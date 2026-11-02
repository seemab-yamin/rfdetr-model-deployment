# First-time setup

Models are not committed to git, so a fresh checkout has no model file. The
app still runs, but `/ping` and `/predict` return 503 and `/health` lists what
is missing in `setup_errors`. Complete this checklist first.

## Checklist
1. Install the CPU dev dependencies: `pip install -r requirements-dev.txt`.
   Never install `requirements-gpu.txt` in the same environment: both ONNX
   Runtime packages provide the `onnxruntime` module.
2. Download RF-DETR Nano to `model.path`:
   `curl -fL -o models/rf-detr-nano.onnx https://huggingface.co/PierreMarieCurie/rf-detr-onnx/resolve/main/rf-detr-nano.onnx`.
   In CI, the workflow downloads `model.url` with `wget` for you.
3. Start the server: `uvicorn app.main:app --reload`. The CPU runtime has no
   `CUDAExecutionProvider`, so the log warns that it is skipped and the
   service runs on CPU.
4. Check `/health`: `model_loaded` is `true` and `model_io` shows input
   `[1, 3, 384, 384]`, outputs `pred_boxes [1, 300, 4]` and
   `pred_logits [1, 300, 91]`.
5. Set the Docker Hub secrets in GitHub (see "CI" below) so CI pushes the GPU image.

## GPU
The Docker image installs `onnxruntime-gpu[cuda,cudnn]`, which brings CUDA 12
and cuDNN 9 as pip wheels. `app/services/onnx_runtime.py` preloads them with
`onnxruntime.preload_dlls()` when `CUDAExecutionProvider` is requested.

Run the image with `docker run --gpus all ...` on a host with an NVIDIA
driver and the NVIDIA Container Toolkit. Then check `/health`:
`model_io.providers` lists the providers the session actually uses.

| `model_io.providers` | Meaning |
|---|---|
| `[CUDAExecutionProvider, CPUExecutionProvider]` | Running on the GPU |
| `[CPUExecutionProvider]` | GPU not visible: missing `--gpus all`, driver too old for CUDA 12, or toolkit not installed. The container logs show the CUDA error |

Fallback order: the loader keeps the providers from `inference.providers` that
this ONNX Runtime build has, in order, and always ends with CPU. ONNX Runtime
then drops CUDA at session start if no GPU is visible. Either way the service
starts, `/ping` turns 200, and `requested_providers` in `/health` shows what
was asked for. To force CPU on a GPU host, set
`INFERENCE_PROVIDERS=CPUExecutionProvider`.

## Switching RF-DETR size
Change `model.path` and `model.url` to another export from
https://huggingface.co/PierreMarieCurie/rf-detr-onnx (use the `resolve/main/`
link, not `blob/main/`). The input size comes from the model, so
`model.input_size` only matters for dynamic inputs. Labels, normalization, and
postprocessing are the same for all COCO detection sizes. Segmentation exports
(`rf-detr-seg-*`) add a mask output that this service ignores.

A fine-tuned RF-DETR with its own classes needs `model.labels` replaced with
its class list, in model output order.

## Config fields
See comments in `config.yaml`. Unknown keys are rejected on purpose.
- `model.url` (direct link): CI downloads the model from here to
  `model.path`. The app itself never downloads it.
- `model.mean`, `model.std`: ImageNet values. RF-DETR was trained with them.
- `inference.providers`: priority order. Unavailable ones are skipped, CPU is the last resort. `INFERENCE_PROVIDERS` overrides it.
- `inference.max_upload_mb` (int, default 10): `/predict` rejects larger uploads with 413.

## Calling /predict
```bash
curl -F "file=@data/sample/bus.jpg" localhost:8000/predict
```
`/annotate` takes the same upload and returns the image with boxes drawn:
`curl -F "file=@data/sample/bus.jpg" localhost:8000/annotate -o annotated.jpg`.

Send one JPEG, PNG, or WebP image in the form field `file`. Until setup is
complete, `/predict` returns 503 with the same `setup_errors` as `/health`.
Fix those first. A 500 with a specific message (for example a labels count
mismatch) means `config.yaml` or `app/pipeline/` needs a change. See the
README for the full error table.

## Mocks in unit tests
Unit tests never need a real model or a GPU.
- `tests/unit/test_onnx_runtime.py` fakes ONNX metadata, sessions, and the CUDA preload.
- `tests/unit/test_startup.py` patches `app.main.load_model` with a fake model.
- `tests/unit/test_detection_pipeline.py` feeds fake RF-DETR outputs.
- `tests/unit/test_annotate.py` uses a fake pipeline for `/annotate`.

Do not add a real .onnx file to the repo to make tests pass.

## CI
The workflow in `.github/workflows/ci.yml` reads `model.url` and
`model.path` from `config.yaml`, downloads the model with `wget`, builds the
GPU image, and runs the integration tests against it. Registry settings go
under Settings > Secrets and variables > Actions:

| Name | Kind | Purpose |
|---|---|---|
| `DOCKERHUB_USERNAME` | secret | Registry username |
| `DOCKERHUB_TOKEN` | secret | Registry access token |
| `MODEL_URL` | repository variable (optional) | Overrides `model.url` without a commit |

GitHub-hosted runners have no GPU, so the integration tests run the GPU image
on ONNX Runtime's CPU fallback. They check endpoints and response shape, not
GPU speed. Pushes to `main` with both secrets set push `<user>/<repo>:<sha>`
and `:latest`. Pull requests never push.

A file that downloads but is not a valid model fails at the `/ping` wait, and
the log prints `/health` with `setup_errors` that say what is wrong. Deploy the
SHA tag, not `latest`.
