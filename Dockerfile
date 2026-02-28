# ============================================================
# Stage 1: dependency builder
# ============================================================
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy only the files needed to resolve & install deps
COPY pyproject.toml .
# uv needs a lockfile; create one during build if it doesn't exist
# (in practice you commit uv.lock to the repo)
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# ============================================================
# Stage 2: runtime image
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtualenv from the builder stage
COPY --from=builder /app/.venv .venv

# Copy source code and pre-trained model
COPY src/ src/
COPY models/ models/

# Activate the virtualenv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "ds_demo.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
