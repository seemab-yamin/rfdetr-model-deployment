import numpy as np

from app.pipeline.base import PipelineError
from app.pipeline.detection.preprocess import LetterboxContext

MAX_DET = 300
CLASS_OFFSET = 7680  # standard trick for class-aware NMS


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(output: np.ndarray):
    """YOLOv8-style output (1, 4+nc, N) or (1, N, 4+nc): cx, cy, w, h, class scores."""
    arr = output[0]
    if arr.ndim != 2 or min(arr.shape) < 5:
        raise PipelineError(
            f"Unexpected detection output shape {output.shape}. The reference expects a "
            "YOLOv8-style layout (1, 4+classes, N). Edit app/pipeline/detection/postprocess.py."
        )
    if arr.shape[0] < arr.shape[1]:  # (C, N) -> (N, C)
        arr = arr.T
    scores = arr[:, 4:]
    return arr[:, :4], scores.max(axis=1), scores.argmax(axis=1), scores.shape[1]


def postprocess(
    output, ctx: LetterboxContext, labels, conf_thr: float, iou_thr: float
) -> list[dict]:
    boxes, conf, cls, num_classes = decode(output)
    if labels and len(labels) != num_classes:
        raise PipelineError(
            f"config.yaml model.labels has {len(labels)} names but the model outputs "
            f"{num_classes} classes."
        )

    mask = conf >= conf_thr
    boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
    if boxes.shape[0] == 0:
        return []

    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    keep = nms(xyxy + cls[:, None] * CLASS_OFFSET, conf, iou_thr)[:MAX_DET]
    xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

    # undo letterbox: remove padding, divide by scale, clip to the original image
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - ctx.pad_x) / ctx.scale).clip(0, ctx.orig_w)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - ctx.pad_y) / ctx.scale).clip(0, ctx.orig_h)

    return [
        {
            "class_id": int(c),
            "class": labels[int(c)] if labels else str(int(c)),
            "confidence": round(float(s), 3),
            "bbox": [round(float(v), 1) for v in box],
        }
        for box, s, c in zip(xyxy, conf, cls)
    ]
