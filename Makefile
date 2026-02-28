.PHONY: setup train serve test lint format docker-build docker-run clean

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_NAME  ?= ds-cicd-demo
IMAGE_TAG   ?= latest
PORT        ?= 7860

# ── Dev setup ─────────────────────────────────────────────────────────────────
setup:  ## Install all dependencies (including dev)
	uv sync --all-groups

# ── ML pipeline ───────────────────────────────────────────────────────────────
train:  ## Train the Iris classifier and save to models/
	uv run python -m ds_demo.models.train

# ── API server ────────────────────────────────────────────────────────────────
serve:  ## Run FastAPI dev server on localhost:8000
	uv run uvicorn ds_demo.api.app:app --host 0.0.0.0 --port 8000 --reload

# ── Tests ─────────────────────────────────────────────────────────────────────
test:  ## Run pytest with coverage report
	uv run pytest tests/ --cov=src --cov-report=term-missing

# ── Linting & formatting ──────────────────────────────────────────────────────
lint:  ## Run ruff linter
	uv run ruff check src tests

format:  ## Auto-format code with ruff
	uv run ruff format src tests

lint-fix:  ## Run ruff linter with auto-fix
	uv run ruff check --fix src tests

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:  ## Build the Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:  ## Run the container locally on port 7860
	docker run --rm -p $(PORT):7860 $(IMAGE_NAME):$(IMAGE_TAG)

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:  ## Remove __pycache__, .pytest_cache, coverage files
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null; true
	rm -f coverage.xml .coverage

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
