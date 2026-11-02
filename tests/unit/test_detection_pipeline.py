from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.pipeline.base import PipelineError
from app.pipeline.detection import DetectionPipeline
from app.pipeline.detection.postprocess import nms, postprocess
from app.pipeline.detection.preprocess import (
    LetterboxContext,
    letterbox,
    resolve_image_spec,
    to_tensor,
)
from app.services.onnx_runtime import TensorSpec

CTX = LetterboxContext(640, 640, 1.0, 0, 0)


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


def test_letterbox_geometry():
    canvas, ctx = letterbox(Image.new("RGB", (200, 100)), 640, 640)
    assert canvas.shape == (640, 640, 3)
    assert (ctx.scale, ctx.pad_x, ctx.pad_y) == (3.2, 0, 160)
    assert canvas[0, 0, 0] == 114


def test_to_tensor_rgb_bgr_layout():
    canvas = np.zeros((4, 4, 3), dtype=np.uint8)
    canvas[..., 0] = 255
    assert to_tensor(canvas, "NCHW", "RGB", True)[0, 0].max() == 1.0
    assert to_tensor(canvas, "NCHW", "BGR", True)[0, 2].max() == 1.0
    assert to_tensor(canvas, "NHWC", "RGB", True).shape == (1, 4, 4, 3)


# postprocess
def make_output(n=10, nc=2):
    return np.zeros((1, 4 + nc, n), dtype=np.float32)


def set_det(out, i, cx, cy, w, h, cls, score):
    out[0, :4, i] = [cx, cy, w, h]
    out[0, 4 + cls, i] = score


def test_postprocess_basic():
    out = make_output()
    set_det(out, 0, 320, 320, 100, 100, 1, 0.9)
    res = postprocess(out, CTX, ["a", "b"], 0.25, 0.45)
    assert len(res) == 1 and res[0]["class"] == "b"
    assert res[0]["bbox"] == [270.0, 270.0, 370.0, 370.0]


def test_postprocess_accepts_transposed_layout():
    out = make_output()
    set_det(out, 0, 320, 320, 100, 100, 0, 0.9)
    assert len(postprocess(out.transpose(0, 2, 1), CTX, [], 0.25, 0.45)) == 1


def test_postprocess_undoes_letterbox():
    out = make_output()
    set_det(out, 0, 320, 320, 64, 64, 0, 0.9)
    ctx = LetterboxContext(200, 100, 3.2, 0, 160)
    bbox = postprocess(out, ctx, [], 0.25, 0.45)[0]["bbox"]
    assert bbox == pytest.approx([90, 40, 110, 60], abs=0.2)


def test_threshold_filters():
    out = make_output()
    set_det(out, 0, 320, 320, 100, 100, 0, 0.1)
    assert postprocess(out, CTX, [], 0.25, 0.45) == []


def test_nms_is_class_aware():
    same = make_output()
    set_det(same, 0, 320, 320, 100, 100, 0, 0.9)
    set_det(same, 1, 322, 320, 100, 100, 0, 0.8)
    assert len(postprocess(same, CTX, [], 0.25, 0.45)) == 1

    diff = make_output()
    set_det(diff, 0, 320, 320, 100, 100, 0, 0.9)
    set_det(diff, 1, 322, 320, 100, 100, 1, 0.8)
    assert len(postprocess(diff, CTX, [], 0.25, 0.45)) == 2


def test_labels_count_mismatch():
    with pytest.raises(PipelineError, match="labels"):
        postprocess(make_output(), CTX, ["only_one"], 0.25, 0.45)


def test_nms_keeps_highest_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    assert nms(boxes, np.array([0.5, 0.9]), 0.45) == [1]
