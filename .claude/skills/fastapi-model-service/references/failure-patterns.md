# Failure Patterns

Check this file first when debugging. After resolving a new class of failure, add a general pattern here, not a one-off incident. Organize by symptom class, with likely causes and a first check.

## Container starts then dies repeatedly
- Likely causes: startup crash, missing dependency, wrong port, model path not found.
- First check: run the image and read the startup logs before changing anything.

## Intermittent 5xx right after deploy
- Likely causes: traffic arriving before the model is loaded.
- First check: does the readiness endpoint return non-200 until loading finishes?

## Serialization errors on responses
- Likely causes: a non-JSON object (model, tensor, device, numpy type) returned from a route.
- First check: return only plain dicts, lists, and primitives.

## Image pull failure
- Likely causes: malformed reference, private repository, wrong tag.
- First check: the reference must be `user/repo:tag`, and the registry must be reachable and public (or credentialed) for the platform.

## Works locally but fails in the container
- Likely causes: hardcoded paths, missing system libraries, environment variables not set, files excluded by .dockerignore.
- First check: run the built image locally or in CI with the same env vars the platform uses.

## CI passes locally but fails in the runner
- Likely causes: shell quoting in workflow steps, weights or assets ignored by git and absent in CI, slower CPU-only runner timing out.
- First check: read the exact failing step and its full log.

## Out-of-memory or out-of-disk errors
- Likely causes: model too large for instance memory, container disk too small for weights and dependencies.
- First check: compare model and dependency size against the instance's memory and disk limits.

## Slow responses
- Likely causes: cold start, oversized request or response payloads, large input images, CPU-only inference.
- First check: measure each stage separately (queue or cold start, inference, response download).

## Tests failing for the wrong reason
- Likely causes: integration tests running without a live server.
- First check: integration tests should skip unless a base URL is set.

## Adding new patterns
Describe the symptom class and the first check. Keep it general. If a specific incident repeats across two projects, promote it here.