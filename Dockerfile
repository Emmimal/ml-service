# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Installing dependencies in a separate stage means they are cached by Docker
# as long as requirements.txt does not change — even if application code does.
# This cuts rebuild time from ~60 s to < 5 s on a warm cache.
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy only the dependency manifest first so Docker can cache this layer.
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
# The runtime image contains no compiler, no pip cache, and no build tools.
# This keeps the final image small (~300 MB vs ~800 MB for a full build image)
# and reduces the attack surface.
FROM python:3.11-slim AS runtime

# Non-root user for security.  Many container registries and security scanners
# fail images that run as root by default.
RUN groupadd --gid 1001 mluser && \
    useradd --uid 1001 --gid mluser --shell /bin/bash --create-home mluser

WORKDIR /app

# Copy compiled packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy application source.
COPY api/        ./api/
COPY model/      ./model/

# Set the model version as a build argument so it can be injected at
# CI build time without modifying the Dockerfile.
ARG MODEL_VERSION=1.0.0
ENV MODEL_VERSION=${MODEL_VERSION} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Health check — Docker will mark the container unhealthy if this fails,
# which surfaces in `docker ps` and triggers replacement in Swarm / ECS.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

USER mluser

EXPOSE 8000

# Use exec form (not shell form) so SIGTERM reaches uvicorn directly
# rather than being swallowed by a shell process.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
