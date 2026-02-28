---
title: DS CICD Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DS CI/CD Demo — Iris Classifier

A **teaching-grade** example project demonstrating industry best practices for:

- 📁 **Project structure** — cookiecutter-inspired `src/` layout
- 🐍 **Python 3.12 + uv** — fast dependency management
- 🐳 **Docker** — multi-stage build, exposed on port 7860 (HF Spaces compatible)
- ⚙️ **GitHub Actions CI/CD** — lint → test → build → deploy pipeline
- 🤗 **Hugging Face Spaces** — free hosting via git push

**Use case:** Iris flower species classifier served via FastAPI.

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/<you>/ds-cicd-demo
cd ds-cicd-demo
make setup          # uv sync --all-groups

# 2. Train the model
make train          # saves models/iris_classifier.joblib

# 3. Run the API
make serve          # http://localhost:8000/docs

# 4. Run tests
make test
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Model info + version |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict Iris species |

### Example prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

```json
{
  "species": "setosa",
  "class_id": 0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```

---

## Project Structure

```
ds-cicd-demo/
├── src/ds_demo/
│   ├── data/           # load + split dataset
│   ├── features/       # feature engineering
│   ├── models/         # train + predict
│   └── api/            # FastAPI app
├── tests/
│   ├── unit/           # feature + model unit tests
│   └── integration/    # API integration tests (TestClient)
├── .github/workflows/
│   ├── ci.yml          # lint + test on every push/PR
│   └── cd.yml          # build + push + deploy on merge to main
├── Dockerfile          # multi-stage, exposes 7860
├── Makefile            # convenience targets
└── pyproject.toml      # uv project file
```

---

## CI/CD Pipeline

```
push feature branch
      │
      ▼
  CI workflow
  ├── ruff lint + format check
  ├── train model
  └── pytest (unit + integration) + coverage

PR merged to main
      │
      ▼
  CD workflow
  ├── train model
  ├── build Docker image
  ├── push to ghcr.io/<user>/ds-cicd-demo
  └── git push → Hugging Face Spaces auto-rebuilds
```

---

## Secrets (GitHub repo settings)

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | Hugging Face write token |
| `HF_SPACE_NAME` | e.g. `username/ds-cicd-demo` |

`GITHUB_TOKEN` is provided automatically by GitHub Actions — no extra secret needed for `ghcr.io`.

---

## Local Docker

```bash
make train          # ensure model exists
make docker-build   # build image
make docker-run     # run on http://localhost:7860
```

---

## Pre-commit Hooks

```bash
uv run pre-commit install
```

Runs `ruff`, `ruff-format`, YAML/TOML checks, and whitespace fixes on every commit.
