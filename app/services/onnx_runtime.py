import logging
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


class ModelLoadError(Exception):
    """Model problem. The message is written for the person setting up the repo."""


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: type | None
    shape: tuple  # ints, strings, or None for dynamic dims


def _spec(meta, strict: bool) -> TensorSpec:
    dtype = _DTYPES.get(meta.type)
    if dtype is None and strict:
        raise ModelLoadError(
            f"Unsupported input type {meta.type} on '{meta.name}'. Supported: {sorted(_DTYPES)}."
        )
    return TensorSpec(meta.name, dtype, tuple(meta.shape))


class OnnxModel:
    """Thin wrapper over an ONNX Runtime session. It knows nothing about tasks."""

    def __init__(self, session, inputs: list[TensorSpec], outputs: list[TensorSpec]):
        self.session = session
        self.inputs = inputs
        self.outputs = outputs

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        expected = {s.name for s in self.inputs}
        if set(feed) != expected:
            raise ValueError(
                f"Model expects inputs {sorted(expected)}, got {sorted(feed)}."
            )
        cast = {
            s.name: np.asarray(feed[s.name]).astype(s.dtype, copy=False)
            for s in self.inputs
        }
        return self.session.run(None, cast)

    def info(self) -> dict:
        return {
            "inputs": [
                {
                    "name": s.name,
                    "dtype": np.dtype(s.dtype).name,
                    "shape": list(s.shape),
                }
                for s in self.inputs
            ],
            "outputs": [{"name": s.name, "shape": list(s.shape)} for s in self.outputs],
            "providers": self.session.get_providers(),
        }


def load_model(model_cfg, inference_cfg) -> OnnxModel:
    available = ort.get_available_providers()
    missing = [p for p in inference_cfg.providers if p not in available]
    if missing:
        raise ModelLoadError(
            f"Providers not available: {missing}. Available here: {available}. "
            "For GPU, install onnxruntime-gpu instead of onnxruntime, "
            "or set inference.providers to [CPUExecutionProvider] in config.yaml."
        )
    try:
        session = ort.InferenceSession(
            str(model_cfg.path), providers=inference_cfg.providers
        )
    except Exception as e:
        raise ModelLoadError(
            f"Could not load {model_cfg.path} as an ONNX model: {e}"
        ) from e

    inputs = [_spec(m, strict=True) for m in session.get_inputs()]
    outputs = [_spec(m, strict=False) for m in session.get_outputs()]
    return OnnxModel(session, inputs, outputs)
