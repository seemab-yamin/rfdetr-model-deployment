---
name: fastapi-model-service
description: Use when building, adapting, or debugging a FastAPI service that serves an ONNX model (vision, NLP, tabular) through a config-driven task pipeline, in a Docker container with GitHub Actions CI. Triggers on "serve a model", "deploy a model API", "swap the model", "add an endpoint", "add a task", "adapt the template", "scaffold a model service", and deployment, CI, or health-check failures.
---

# Workflow
1. Read the repo structure, README, and `docs/first-time-setup.md` before changing anything.
2. If the request does not state the model, input type, and output shape, ask. Otherwise proceed.
3. Change only the swap points below.
4. Run `pytest tests/unit`, then run the app locally and call `/health` and `/ping`.
5. Confirm the Dockerfile still copies `models/` and downloads nothing. CI downloads the model before the build.
6. Report: files changed, why, and what to verify in CI.

# Design rule
When a design choice is open, follow the most common convention in the field. Do not invent.

# Swap points (edit these only)
- `config.yaml`: `model.path`, `model.url`, `model.labels`, `model.input_size`, `model.channel_order`, `model.normalize`, and the `inference` thresholds
- `app/pipeline/<task>/preprocess.py`: input handling (resize, color order, scaling)
- `app/pipeline/<task>/postprocess.py`: turning raw model outputs into the JSON result
- Model download URL: `model.url` in `config.yaml` (the `MODEL_URL` CI repository variable overrides it). It must be a direct file link
- `app/schemas/`: response shapes, only when the output shape changes
- `requirements.txt`: runtime dependencies only. Test tools go in `requirements-dev.txt`
- `tests/`: update tests and fakes to match the new output shape

# Adding a new task (NLP, tabular, other)
1. Create `app/pipeline/<task>/` with a `Pipeline` subclass (see `app/pipeline/base.py`) that implements `preprocess` and `postprocess`.
2. Register it in the `PIPELINES` registry in `app/pipeline/__init__.py`.
3. Add the task name to the `task` Literal in `app/config.py`.
4. Set `model.task` in `config.yaml`.
5. Add unit tests that feed fake arrays and check the output.

# Adding an endpoint
1. New file in `app/routers/`, registered in `app/main.py`.
2. Model logic stays in `app/pipeline/`. Request helpers go in `app/utils/`. The router stays thin.
3. Request and response Pydantic models in `app/schemas/`.
4. Read everything from `request.app.state`. Return 503 with `setup_errors` while the model is not loaded.
5. Unit test and integration test for it.

# Never change without asking
- Lifespan loading and the `app.state` pattern
- The setup-errors startup flow: startup never crashes on bad setup, problems are collected in `app.state.setup_errors`, `/health` lists them, and model endpoints return 503 with them
- `app/services/onnx_runtime.py`: task-agnostic. Never edit it for a new task
- Readiness logic: `/ping` returns 503 until the model is loaded
- `/health` stays informational (status, model_loaded, model path, model I/O, setup errors)
- Workflow step order, its `if:` gating, and SHA plus `latest` tagging
- Base image and `CMD` in the Dockerfile

If a request requires changing one of these, stop and explain why before editing.

# Architecture rules
- Load the model once in lifespan, never per request.
- Routers stay thin. `app/pipeline/` and `app/services/` never import FastAPI.
- Blocking inference runs in a plain `def` route or `run_in_threadpool`, never directly in an `async def`.
- Never return the model object or any non-JSON type in a response.
- Validate inputs: file type, size limit (`inference.max_upload_mb`), clear 4xx errors.
- Return JSON with stable field names, documented by Pydantic models.
- Unknown config keys are rejected. Add new keys to `app/config.py` and comment them in `config.yaml`.
- Never commit model files. CI downloads them with `wget` from `model.url`.
- No secrets in the image. Use env vars and CI secrets.
- Keep dependencies minimal. Image size drives cold start time.

# Definition of done
- `pytest tests/unit` passes with no real model file
- With a real model: `/health` reports `model_loaded: true` and `/ping` returns 200
- With no model: `/ping` and model endpoints return 503, and `/health` lists `setup_errors`
- Integration tests exist for every new endpoint and skip unless `BASE_URL` is set
- README lists curl examples for every endpoint
- CI is green, and its job summary explains any skipped stage

# References (read only when relevant)
- `docs/first-time-setup.md`: read when setting up a new repo, changing config, or adding a task
- `references/failure-patterns.md`: read when debugging
- `references/deployment.md`: read when preparing or changing deployment or CI
