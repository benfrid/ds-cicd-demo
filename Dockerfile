# ============================================================
# Stage 1: dependency builder
# ============================================================
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lockfile + manifest — but NOT src/ yet.
# --no-install-project installs only third-party deps, skipping the local
# package build (which needs src/ and hatchling). This keeps the layer cache
# valid even when application code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

# ============================================================
# Stage 2: runtime image
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the pre-built virtualenv (third-party deps only)
COPY --from=builder /app/.venv .venv

# Copy source code
COPY src/ src/

# Activate the virtualenv
ENV PATH="/app/.venv/bin:$PATH"
# Make ds_demo importable without a formal pip install
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Train the model at build time — bakes it into the image.
# This avoids committing binary .joblib files to git.
RUN python -m ds_demo.models.train

# HF Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "ds_demo.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
