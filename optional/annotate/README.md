# Optional: `/annotate`

Runs the detection pipeline on an uploaded image and streams the same image
back, in the same format, with boxes and labels drawn on it. Nothing is
saved to disk, so there is no download route and no storage config.

Not part of the core service. `optional/` is excluded from the Docker image
(see `.dockerignore`) until you copy the route into `app/`.

## Enable it
1. Copy the route into the app:

```bash
cp optional/annotate/annotate.py app/routers/annotate.py
```

2. Register it in `app/main.py`:

```python
from app.routers import annotate, health, predict

app.include_router(annotate.router)
```

3. Restart the server. `/annotate` appears in `/docs`.

## Call it

```bash
curl -F "file=@path/to/image.jpg" localhost:8000/annotate -o annotated.jpg
```

The response body is the annotated image, with `Content-Type` matching the
upload (`image/jpeg`, `image/png`, or `image/webp`). The
`X-Detections-Count` header holds the number of boxes. For the detections
as JSON, call `/predict`.

## Errors
Same as `/predict`: 400 (empty, corrupt, or too small image), 413 (over
`inference.max_upload_mb`), 415 (unsupported type), 422 (no `file` field),
500 (pipeline or internal error), 503 (model not loaded, with
`setup_errors`).

## Requirements
Uses the detection result shape: `class`, `confidence`, and `bbox` as
`[x1, y1, x2, y2]` in original-image pixels. If you change the pipeline
output, update `draw_detections` to match. Needs no extra dependencies
beyond Pillow, which the core already uses.
