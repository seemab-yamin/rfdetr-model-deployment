# rfdetr-model-deployment

A GPU REST API for [RF-DETR](https://github.com/roboflow/rf-detr) object detection. It runs the RF-DETR Nano COCO model (ONNX export from [PierreMarieCurie/rf-detr-onnx](https://huggingface.co/PierreMarieCurie/rf-detr-onnx)) with ONNX Runtime CUDA, behind FastAPI, in a Docker image that CI builds, tests, and pushes to Docker Hub.

- `/predict` takes an image and returns COCO detections with boxes in original-image pixels
- `/annotate` returns the same image with the boxes and labels drawn on it
- The model is downloaded by CI and baked into the image, then loaded once at startup
- Startup never crashes on bad setup: problems are listed in `/health`, and `/ping` stays 503 until the model is loaded
- Docker image runs on NVIDIA GPUs (CUDA 12 and cuDNN 9 come as pip wheels, no CUDA base image needed), and falls back to the next available execution provider, ending with CPU, when no GPU is present
- Images tagged with the commit SHA and `latest`

## Run with Docker (GPU)

The host needs an NVIDIA driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker run --gpus all -p 8000:8000 <dockerhub-user>/rfdetr-model-deployment:<sha>
```

Check that the GPU is in use. `model_io.providers` in `/health` should start with `CUDAExecutionProvider`:

```bash
curl http://localhost:8000/health
```

If it only lists `CPUExecutionProvider`, the container cannot see the GPU (missing `--gpus all`, driver, or toolkit). The service still starts and serves on CPU, and the container log has a warning saying why. `requested_providers` shows what `config.yaml` asked for.

To build the image yourself:

```bash
curl -fL -o models/rf-detr-nano.onnx https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx
```

```bash
docker build -t rfdetr-model-deployment .
```

## Local development (CPU)

`onnxruntime-gpu` has no macOS build, so local development and tests use the CPU runtime from `requirements-dev.txt`.

```bash
pip install -r requirements-dev.txt
```

```bash
curl -fL -o models/rf-detr-nano.onnx https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx
```

`config.yaml` asks for CUDA first. The CPU runtime does not have it, so the service logs a warning, skips it, and runs on CPU:

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` and try `/predict` with an image from `data/sample/`. Run the unit tests (no model file needed):

```bash
pytest tests/unit -v
```

Without a model file the server still starts: `/ping` and `/predict` return 503, and `/health` lists what to fix. See `docs/first-time-setup.md`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and info. Returns 200 when the process is up, with `status`, `model_loaded`, `model_path`, `model_io` (input and output shapes, active `providers`, and the `requested_providers` from config), and `setup_errors`. |
| GET | `/ping` | Readiness. Returns 503 until the model is loaded, then 200. Point platform health checks here. |
| POST | `/predict` | Runs RF-DETR on an uploaded image (`multipart/form-data`, field `file`) and returns JSON. |
| POST | `/annotate` | Same input as `/predict`. Returns the image with boxes and labels drawn on it. |

```bash
curl http://localhost:8000/health
```

```bash
curl http://localhost:8000/ping
```

### `/predict`

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@data/sample/bus.jpg"
```

Response (trimmed to two of five detections):

```json
{
  "filename": "bus.jpg",
  "image": {"width": 810, "height": 1080},
  "count": 2,
  "detections": [
    {"class_id": 6, "class": "bus", "confidence": 0.953, "bbox": [9.4, 229.4, 806.8, 734.0]},
    {"class_id": 1, "class": "person", "confidence": 0.948, "bbox": [50.6, 395.0, 245.7, 907.6]}
  ],
  "inference_ms": 64.9
}
```

- `bbox` is `[x1, y1, x2, y2]` in original-image pixels.
- `class_id` is the COCO category id (1 = person, 6 = bus), as RF-DETR outputs it.
- Detections are sorted by confidence. RF-DETR makes one prediction per query (300 queries), so there is no NMS step.
- No detections returns `count: 0` and an empty list with status 200. `inference_ms` covers the model pipeline only (CPU timing above).

| Status | Cause |
|---|---|
| 400 | Empty file, corrupt or truncated image, or a side under 16 px |
| 413 | File larger than `inference.max_upload_mb` (default 10) |
| 415 | Content type or real file format is not `image/jpeg`, `image/png`, or `image/webp` |
| 422 | No `file` field in the form |
| 500 | Pipeline error (message says what to fix, for example a labels count mismatch) or an internal error (generic message, traceback in the server log) |
| 503 | Model not loaded. The body has `setup_errors` listing what to fix |

### `/annotate`

Runs the same detection and returns the uploaded image, in the same format, with boxes and `class confidence` labels drawn on it. Nothing is saved to disk.

```bash
curl -X POST "http://localhost:8000/annotate" -F "file=@data/sample/bus.jpg" -o annotated.jpg
```

The response `Content-Type` matches the upload (`image/jpeg`, `image/png`, or `image/webp`), and the `X-Detections-Count` header holds the number of boxes. Errors are the same as `/predict`. For the detections as JSON, call `/predict`.

## How inference works

1. **Preprocess** (`app/pipeline/detection/preprocess.py`): convert to RGB, resize to 384×384 without padding, scale to 0–1, subtract the ImageNet mean and divide by std, NCHW float32.
2. **Model**: outputs `pred_boxes` `(1, 300, 4)` as normalized cx, cy, w, h, and `pred_logits` `(1, 300, 91)`.
3. **Postprocess** (`app/pipeline/detection/postprocess.py`): sigmoid on the logits, best class per query, keep scores at or above `confidence_threshold`, sort by score, scale boxes to original-image pixels.

## Configuration

Settings live in `config.yaml`. Each key is commented there, and unknown keys are rejected so typos fail loudly.

| Key | Value | Meaning |
|---|---|---|
| `model.path` | `models/rf-detr-nano.onnx` | Where the app loads the model. Must be under `models/` for the image |
| `model.url` | RF-DETR Nano on Hugging Face | Direct link CI downloads to `model.path`. The app never downloads it |
| `model.task` | `detection` | Pipeline in `app/pipeline/` |
| `model.input_size` | `[384, 384]` | Height, width. Used only if the model input is dynamic |
| `model.channel_order` | `RGB` | `RGB` or `BGR` |
| `model.normalize` | `true` | Divide pixel values by 255 |
| `model.mean` | ImageNet mean | Per-channel mean subtracted after `normalize` |
| `model.std` | ImageNet std | Per-channel std divided after `mean` |
| `model.labels` | 91 COCO names | Indexed by COCO category id, `N/A` for unused ids |
| `inference.providers` | `[CUDAExecutionProvider, CPUExecutionProvider]` | ONNX Runtime execution providers, in priority order. Ones this build lacks are skipped, and CPU is always the last resort |
| `inference.confidence_threshold` | `0.5` | Minimum score to keep a detection |
| `inference.max_upload_mb` | `10` | `/predict` rejects larger uploads with 413 |

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | Port the server listens on in the container |
| `CONFIG_PATH` | `config.yaml` | Path to the config file |
| `INFERENCE_PROVIDERS` | unset | Comma-separated override for `inference.providers`, for example `CPUExecutionProvider` to force CPU on a GPU host |

Secrets (API keys, registry tokens) come from env vars or CI secrets. Never commit them and never bake them into the image.

### Another RF-DETR size

The same Hugging Face repo has Small, Medium, Base, and larger exports. To switch, change `model.path` and `model.url` in `config.yaml`, and check `/health` `model_io` for the input size (for example Base is 560×560). Fixed model input sizes override `model.input_size` automatically. The COCO labels stay the same. Note that `rf-detr-xlarge` and `rf-detr-2xlarge` use Roboflow's PML-1.0 license, not Apache-2.0.

## Project structure

```
app/
  main.py          # lifespan, app.state, setup_errors, router registration
  config.py        # loads and validates config.yaml (unknown keys rejected)
  routers/         # thin HTTP layer (health.py, predict.py, annotate.py)
  pipeline/        # detection/ (preprocess.py, postprocess.py) for RF-DETR
  services/        # onnx_runtime.py: ONNX Runtime session, provider fallback, CUDA library preload
  schemas/         # request and response models
  utils/           # upload and image validation, box drawing
config.yaml        # model and inference settings
models/            # .onnx goes here (git keeps only .gitkeep)
tests/
  unit/            # run on every push, no model needed
  integration/     # run against a live container, skip unless BASE_URL is set
docs/first-time-setup.md
Dockerfile
requirements.txt       # shared runtime dependencies
requirements-gpu.txt   # + onnxruntime-gpu with CUDA/cuDNN wheels (Docker image)
requirements-dev.txt   # + CPU onnxruntime and test tools (local dev, CI unit tests)
.github/workflows/ci.yml
```

## CI/CD

The workflow (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request:

1. Installs the CPU dependencies and runs unit tests
2. Reads `model.url` and `model.path` from `config.yaml`, downloads the model with `wget`, and builds the GPU Docker image
3. Starts the container, waits for `/ping`, and runs integration tests against it
4. Pushes the image to Docker Hub, tagged with the commit SHA and `latest` (push to `main` only, never on pull requests)

GitHub-hosted runners have no GPU, so step 3 tests the GPU image on its CPU fallback, which also proves the no-GPU path works. It checks the endpoints and response shape, not GPU speed. The job summary lists the active providers. Check `/health` on your GPU host after deploying.

Registry settings go under **Settings > Secrets and variables > Actions**:

| Name | Kind | Value |
|---|---|---|
| `DOCKERHUB_USERNAME` | secret | Your registry username |
| `DOCKERHUB_TOKEN` | secret | A registry access token with Read & Write access |
| `MODEL_URL` | repository variable (optional) | Overrides `model.url`, for example to test another RF-DETR size without a commit |

| Configured | What runs |
|---|---|
| `model.url`, no secrets | Unit tests, download, build, integration tests. Push is skipped. |
| `model.url` and both secrets, push to `main` | Everything, and the image is pushed as `<username>/<repo>:<sha>` and `:latest`. |

The job summary lists each stage's result, where the model came from, and why any stage was skipped. The image name is `<DOCKERHUB_USERNAME>/<GitHub repo name>`, lowercased. Docker Hub creates the repository on the first push.

## Deploy

Any platform that runs containers with NVIDIA GPUs works:

- Deploy the pinned SHA tag, not `latest`
- Give the container GPU access (`--gpus all`, or the platform's GPU setting)
- Point the platform health check at `/ping`
- Confirm `/health` lists `CUDAExecutionProvider` first after deploy
- Roll back by redeploying an earlier SHA tag

### RunPod (Serverless, load-balancing endpoint)

The service runs on a RunPod Serverless **load-balancing** endpoint. RunPod forwards plain HTTP to the FastAPI container, so `/predict`, `/annotate`, `/health`, and `/docs` work unchanged. No RunPod handler code is needed.

| Setting | Value |
|---|---|
| Image | `<dockerhub-user>/rfdetr-model-deployment:<sha>` (pushed by CI from `main`) |
| Endpoint type | Load balancing |
| HTTP port | `8000` |
| Health check | `/ping` on port `8000` (503 until the model loads, then 200) |
| Workers | 2 |
| Idle timeout | 60 s |
| Request timeout | 60 s |
| FlashBoot | Off |
| Container disk | 10 GB (the image carries CUDA and cuDNN) |

Every request needs a RunPod API key as a Bearer token. RunPod checks it at the gateway, before the request reaches the app:

```bash
curl -H "Authorization: Bearer $RUNPOD_API_KEY" https://<endpoint-id>.api.runpod.ai/health
```

```bash
curl -H "Authorization: Bearer $RUNPOD_API_KEY" -F "file=@data/sample/bus.jpg" https://<endpoint-id>.api.runpod.ai/predict
```

```bash
curl -H "Authorization: Bearer $RUNPOD_API_KEY" -F "file=@data/sample/bus.jpg" https://<endpoint-id>.api.runpod.ai/annotate -o annotated.jpg
```

After each deploy, check that `/health` → `model_io.providers` starts with `CUDAExecutionProvider`. If it lists only `CPUExecutionProvider`, the worker is not using the GPU (the service still answers, slowly).

> **Tip: open `/docs` in the browser with ModHeader.** A browser cannot send the `Authorization` header, so RunPod rejects `/docs` and the `openapi.json` it loads. Install the ModHeader extension (Chrome or Firefox), add the request header `Authorization: Bearer <your RunPod API key>`, and set its URL filter to `https://<endpoint-id>.api.runpod.ai/*` so the key goes nowhere else. Then open `https://<endpoint-id>.api.runpod.ai/docs`. "Try it out" works too, since ModHeader adds the header to those calls. Swagger's own "Authorize" button does not help here: the gateway blocks the page before it loads. Turn the header off when you are done.

Notes from the first deploy:

- **Keep API keys out of files.** Use an env var such as `RUNPOD_API_KEY`. Rotate any key that was written to a file or pasted somewhere.
- **Cold starts.** With FlashBoot off and workers scaling down after 60 s idle, the first request after idle waits for a worker to pull the multi-GB image and load the model. Keep the 60 s request timeout in mind for that first call, or turn on FlashBoot if cold starts matter.
- **Model URL.** Use Hugging Face `resolve/main/...` links. `blob/main/...` downloads an HTML page, and CI's download check fails.
- **Rollback.** Point the endpoint at an earlier SHA tag.

## Known limits

- One model per service.
- No authentication. Add it before exposing the service publicly.
- Inference runs synchronously per request, one image at a time. Add a queue or batching for high load.
- The GPU image is several GB because it carries CUDA and cuDNN.

## License

MIT. See `LICENSE`. RF-DETR Nano and its weights are Apache-2.0 (Roboflow). The ONNX export is from [PierreMarieCurie/rf-detr-onnx](https://github.com/PierreMarieCurie/rf-detr-onnx) (MIT).
