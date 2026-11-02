import numpy as np

from app.pipeline.base import PipelineError
from app.pipeline.detection.preprocess import ResizeContext

MAX_DET = 300


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def decode(outputs: list[np.ndarray]):
    """RF-DETR / DETR-style outputs: boxes (1, Q, 4) as normalized cx, cy, w, h,
    then class logits (1, Q, classes). One prediction per query, no NMS."""
    if len(outputs) < 2:
        raise PipelineError(
            f"The detection pipeline expects 2 model outputs (boxes, logits), got {len(outputs)}. "
            "Edit app/pipeline/detection/postprocess.py."
        )
    boxes, logits = outputs[0], outputs[1]
    if (
        boxes.ndim != 3
        or boxes.shape[-1] != 4
        or logits.ndim != 3
        or logits.shape[:2] != boxes.shape[:2]
    ):
        raise PipelineError(
            f"Unexpected detection output shapes {boxes.shape} and {logits.shape}. The reference "
            "expects an RF-DETR-style layout: boxes (1, queries, 4) and logits (1, queries, classes). "
            "Edit app/pipeline/detection/postprocess.py."
        )
    prob = sigmoid(logits[0])
    return boxes[0], prob.max(axis=1), prob.argmax(axis=1), prob.shape[1]


def postprocess(outputs, ctx: ResizeContext, labels, conf_thr: float) -> list[dict]:
    boxes, conf, cls, num_classes = decode(outputs)
    if labels and len(labels) != num_classes:
        raise PipelineError(
            f"config.yaml model.labels has {len(labels)} names but the model outputs "
            f"{num_classes} classes."
        )

    mask = conf >= conf_thr
    boxes, conf, cls = boxes[mask], conf[mask], cls[mask]
    if boxes.shape[0] == 0:
        return []

    order = conf.argsort()[::-1][:MAX_DET]
    boxes, conf, cls = boxes[order], conf[order], cls[order]

    # normalized cx, cy, w, h -> x1, y1, x2, y2 in original-image pixels
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] * ctx.orig_w).clip(0, ctx.orig_w)
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] * ctx.orig_h).clip(0, ctx.orig_h)

    return [
        {
            "class_id": int(c),
            "class": labels[int(c)] if labels else str(int(c)),
            "confidence": round(float(s), 3),
            "bbox": [round(float(v), 1) for v in box],
        }
        for box, s, c in zip(xyxy, conf, cls)
    ]
