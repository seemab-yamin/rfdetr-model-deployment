from types import SimpleNamespace

import numpy as np
import pytest

from app.services import onnx_runtime as rt
from app.services.onnx_runtime import ModelLoadError, TensorSpec

@pytest.mark.parametrize(
    "requested, available, expected",
    [
        # CPU-only build: CUDA skipped
        (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"], ["CPUExecutionProvider"]),
        # GPU build: order kept
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
        # nothing requested is available: CPU is the last resort
        (["CUDAExecutionProvider"], ["CoreMLExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
        # next available option is used before CPU
        (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ),
    ],
    ids=["cpu-build", "gpu-build", "none-available", "next-option"],
)
def test_resolve_providers_falls_back(requested, available, expected):
    assert rt.resolve_providers(requested, available) == expected


def test_missing_gpu_does_not_block_loading(monkeypatch, tmp_path, caplog):
    seen = {}

    class Session:
        def __init__(self, path, providers):
            seen["providers"] = providers

        def get_inputs(self):
            return []

        def get_outputs(self):
            return []

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(rt.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(rt.ort, "InferenceSession", Session)
    requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    model = rt.load_model(SimpleNamespace(path=tmp_path / "m.onnx"), SimpleNamespace(providers=requested))

    assert seen["providers"] == ["CPUExecutionProvider"]
    assert model.info()["providers"] == ["CPUExecutionProvider"]
    assert model.info()["requested_providers"] == requested
    assert "CUDAExecutionProvider" in caplog.text


@pytest.mark.parametrize(
    "providers, expected",
    [
        (["CUDAExecutionProvider", "CPUExecutionProvider"], 1),
        (["CPUExecutionProvider"], 0),
    ],
)
def test_preloads_cuda_libs_only_for_cuda(monkeypatch, tmp_path, providers, expected):
    calls = []
    monkeypatch.setattr(rt.ort, "get_available_providers", lambda: providers)
    monkeypatch.setattr(rt.ort, "preload_dlls", lambda: calls.append(1), raising=False)
    bad = tmp_path / "m.onnx"
    bad.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError, match="Could not load"):
        rt.load_model(SimpleNamespace(path=bad), SimpleNamespace(providers=providers))
    assert len(calls) == expected


def test_preload_failure_does_not_block_loading(monkeypatch, tmp_path):
    def boom():
        raise OSError("libcudnn.so.9: cannot open shared object file")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setattr(rt.ort, "get_available_providers", lambda: providers)
    monkeypatch.setattr(rt.ort, "preload_dlls", boom, raising=False)
    bad = tmp_path / "m.onnx"
    bad.write_bytes(b"not a model")
    # gets past preloading to the real load step
    with pytest.raises(ModelLoadError, match="Could not load"):
        rt.load_model(SimpleNamespace(path=bad), SimpleNamespace(providers=providers))


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
