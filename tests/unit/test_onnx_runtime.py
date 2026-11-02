from types import SimpleNamespace

import numpy as np
import pytest

from app.services import onnx_runtime as rt
from app.services.onnx_runtime import ModelLoadError, TensorSpec

def test_missing_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rt.ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    model_cfg = SimpleNamespace(path=tmp_path / "m.onnx")
    inf_cfg = SimpleNamespace(providers=["CUDAExecutionProvider"])
    with pytest.raises(ModelLoadError, match="not available"):
        rt.load_model(model_cfg, inf_cfg)


def test_invalid_onnx_file(tmp_path):
    bad = tmp_path / "m.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError, match="Could not load"):
        rt.load_model(
            SimpleNamespace(path=bad),
            SimpleNamespace(providers=["CPUExecutionProvider"]),
        )


def test_unsupported_input_type():
    meta = SimpleNamespace(name="x", shape=[1], type="tensor(string)")
    with pytest.raises(ModelLoadError, match="Unsupported"):
        rt._spec(meta, strict=True)


class FakeSession:
    def run(self, names, feed):
        self.feed = feed
        return [np.zeros((1, 5))]

    def get_providers(self):
        return ["CPUExecutionProvider"]


def make_model(session=None):
    inputs = [
        TensorSpec("input_ids", np.int64, ("batch", "seq")),
        TensorSpec("attention_mask", np.int64, ("batch", "seq")),
    ]
    return rt.OnnxModel(
        session or FakeSession(), inputs, [TensorSpec("out", None, (1, 5))]
    )


def test_multi_input_and_dtype_cast():
    session = FakeSession()
    model = make_model(session)
    model.run({"input_ids": np.ones((1, 4)), "attention_mask": np.ones((1, 4))})
    assert session.feed["input_ids"].dtype == np.int64


def test_wrong_input_names_rejected():
    with pytest.raises(ValueError, match="expects inputs"):
        make_model().run({"wrong": np.zeros((1, 4))})


def test_info_is_json_safe():
    import json

    json.dumps(make_model().info())
