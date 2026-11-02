# Deployment

Platform-agnostic requirements for deploying a model service built from this template. Platform-specific quirks belong in the project's README, not here.

## Scope
- CI downloads the model, builds, tests, and pushes the image to a registry, then stops.
- Deploying the image to a platform is a per-project step.

## Port
- The app listens on `0.0.0.0`.
- The port comes from the `PORT` env var, default `8000`.

## Health endpoints
- `/health` (liveness and info): returns 200 whenever the process is up, with status, model_loaded, model path, model I/O, and setup_errors. For humans and monitoring.
- `/ping` (readiness): returns 503 until the model is loaded, then 200. Platform health checks and load balancers target this.
- Do not merge them. A single endpoint either reports errors during normal startup or routes traffic before the model is ready.
- If a platform requires a different health path, add an alias route in that project. Do not rename the template's routes.

## Image
- Bake model weights into the image at build time.
- Models are not committed to git. CI downloads `model.url` from `config.yaml` with `wget` into `model.path` under `models/`, checks the file, then builds. The Dockerfile itself downloads nothing.
- `model.url` must be a direct, public file link. Authenticated downloads are out of scope.
- Keep secrets out of the image.
- Tag every build with the commit SHA and `latest`.
- Deploy using the pinned SHA tag, not `latest`.
- Roll back by redeploying a previous SHA tag.

## Config
- All environment-specific values come from env vars.
- Secrets come from the platform's secret store or CI secrets, never from the repo.
- Document every required env var in the README.

## Scaling and cost
- Prefer scale-to-zero for demos and low traffic.
- Cap the maximum number of instances.
- Set request and execution timeouts.

## Hardware
- Use CPU for small models and demos.
- Move to GPU only when measured latency requires it.

## Pre-deploy checklist
- Unit and integration tests pass.
- The image runs locally or in CI.
- `/health` and `/ping` respond as described above.
- Env vars are documented.
- The rollback tag is noted.

## Post-deploy check
- Call `/ping` and `/health` on the live endpoint.
- Send one real request to the main endpoint and confirm the response.

## Platform notes
- Record platform quirks (port variables, required health paths, image reference format) in the project's README.
- If a quirk repeats across projects, add a general rule to this file.