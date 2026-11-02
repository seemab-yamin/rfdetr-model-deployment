from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.pipeline.base import PipelineError
from app.pipeline.detection import DetectionPipeline
from app.pipeline.detection.postprocess import postprocess
from app.pipeline.detection.preprocess import (
    ResizeContext,
    resize,
    resolve_image_spec,
    to_tensor,
)
from app.services.onnx_runtime import TensorSpec

CTX = ResizeContext(640, 480)


def tensor(shape):
    return SimpleNamespace(name="images", shape=shape)


# preprocess
def test_spec_fixed_nchw_uses_model_size():
    s = resolve_image_spec(tensor([1, 3, 320, 320]), (640, 640))
    assert (s.height, s.width, s.layout) == (320, 320, "NCHW")


def test_spec_dynamic_uses_config_size():
    s = resolve_image_spec(tensor(["batch", 3, "h", "w"]), (640, 480))
    assert (s.height, s.width) == (640, 480)


def test_spec_nhwc_detected():
    assert resolve_image_spec(tensor([1, 224, 224, 3]), (640, 640)).layout == "NHWC"


def test_spec_rejects_non_4d():
    with pytest.raises(PipelineError, match="4D"):
        resolve_image_spec(tensor([1, 128]), (640, 640))


def test_pipeline_rejects_multi_input_model():
    model = SimpleNamespace(
        inputs=[
            TensorSpec("a", np.float32, (1, 3, 64, 64)),
            TensorSpec("b", np.float32, (1,)),
        ]
    )
    with pytest.raises(PipelineError, match="1 model input"):
        DetectionPipeline(
            SimpleNamespace(input_size=(640, 640)), SimpleNamespace(), model
        )


def test_resize_stretches_without_padding():
    canvas, ctx = resize(Image.new("RGB", (200, 100), (10, 20, 30)), 384, 384)
    assert canvas.shape == (384, 384, 3)
    assert (ctx.orig_w, ctx.orig_h) == (200, 100)
    assert tuple(canvas[0, 0]) == (10, 20, 30)  # no gray padding


def test_to_tensor_rgb_bgr_layout():
    canvas = np.zeros((4, 4, 3), dtype=np.uint8)
    canvas[..., 0] = 255
    assert to_tensor(canvas, "NCHW", "RGB", True)[0, 0].max() == 1.0
    assert to_tensor(canvas, "NCHW", "BGR", True)[0, 2].max() == 1.0
    assert to_tensor(canvas, "NHWC", "RGB", True).shape == (1, 4, 4, 3)


def test_to_tensor_applies_mean_std():
    canvas = np.full((2, 2, 3), 255, dtype=np.uint8)
    out = to_tensor(canvas, "NCHW", "RGB", True, (0.5, 0.5, 0.5), (0.25, 0.5, 1.0))
    assert out[0, :, 0, 0] == pytest.approx([2.0, 1.0, 0.5])


# postprocess
def make_outputs(q=10, nc=3):
    boxes = np.zeros((1, q, 4), dtype=np.float32)
    logits = np.full((1, q, nc), -10.0, dtype=np.float32)  # sigmoid ~ 0
    return [boxes, logits]


def set_det(outs, i, cx, cy, w, h, cls, logit):
    outs[0][0, i] = [cx, cy, w, h]
    outs[1][0, i, cls] = logit


def test_postprocess_basic():
    outs = make_outputs()
    set_det(outs, 0, 0.5, 0.5, 0.25, 0.5, 1, 3.0)
    res = postprocess(outs, CTX, ["a", "b", "c"], 0.5)
    assert len(res) == 1 and res[0]["class"] == "b" and res[0]["class_id"] == 1
    assert res[0]["confidence"] == pytest.approx(0.953, abs=1e-3)  # sigmoid(3)
    assert res[0]["bbox"] == [240.0, 120.0, 400.0, 360.0]


def test_boxes_clipped_to_image():
    outs = make_outputs()
    set_det(outs, 0, 0.05, 0.95, 0.2, 0.2, 0, 3.0)
    assert postprocess(outs, CTX, [], 0.5)[0]["bbox"] == [0.0, 408.0, 96.0, 480.0]


def test_sorted_by_confidence_without_nms():
    outs = make_outputs()
    set_det(outs, 0, 0.5, 0.5, 0.2, 0.2, 0, 1.0)
    set_det(outs, 1, 0.5, 0.5, 0.2, 0.2, 0, 3.0)  # same box, same class: both kept
    res = postprocess(outs, CTX, [], 0.5)
    assert [r["confidence"] for r in res] == sorted(
        (r["confidence"] for r in res), reverse=True
    )
    assert len(res) == 2


def test_threshold_filters():
    outs = make_outputs()
    set_det(outs, 0, 0.5, 0.5, 0.2, 0.2, 0, -1.0)  # sigmoid ~ 0.27
    assert postprocess(outs, CTX, [], 0.5) == []


def test_labels_count_mismatch():
    with pytest.raises(PipelineError, match="labels"):
        postprocess(make_outputs(), CTX, ["only_one"], 0.5)


@pytest.mark.parametrize(
    "outs",
    [
        [np.zeros((1, 10, 4))],
        [np.zeros((1, 6, 10)), np.zeros((1, 10, 3))],
        [np.zeros((1, 10, 4)), np.zeros((1, 8, 3))],
    ],
    ids=["one-output", "yolo-layout", "query-mismatch"],
)
def test_rejects_non_detr_layout(outs):
    with pytest.raises(PipelineError, match="postprocess.py"):
        postprocess(outs, CTX, [], 0.5)
