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

    def __init__(
        self,
        session,
        inputs: list[TensorSpec],
        outputs: list[TensorSpec],
        requested_providers: list[str] | None = None,
    ):
        self.session = session
        self.inputs = inputs
        self.outputs = outputs
        self.requested_providers = requested_providers

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
            # active ones first; requested shows what config asked for before fallback
            "providers": self.session.get_providers(),
            "requested_providers": self.requested_providers,
        }


def resolve_providers(requested: list[str], available: list[str]) -> list[str]:
    """Keep the requested providers this ONNX Runtime build has, in order.
    CPU is the last resort, so a missing GPU never stops the service."""
    providers = [p for p in requested if p in available]
    skipped = [p for p in requested if p not in available]
    if skipped:
        logger.warning(
            "Providers not in this ONNX Runtime build, skipping: %s. Available: %s. "
            "For GPU, install onnxruntime-gpu instead of onnxruntime.",
            skipped,
            available,
        )
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    return providers


def load_model(model_cfg, inference_cfg) -> OnnxModel:
    providers = resolve_providers(inference_cfg.providers, ort.get_available_providers())
    # onnxruntime-gpu ships CUDA and cuDNN as pip wheels; load them before the session.
    # No-op on CPU builds and macOS. A failure here only means CUDA falls back to CPU.
    if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:
            logger.warning("Could not preload CUDA/cuDNN libraries: %s", e)
    try:
        session = ort.InferenceSession(
            str(model_cfg.path), providers=providers
        )
    except Exception as e:
        raise ModelLoadError(
            f"Could not load {model_cfg.path} as an ONNX model: {e}"
        ) from e

    inputs = [_spec(m, strict=True) for m in session.get_inputs()]
    outputs = [_spec(m, strict=False) for m in session.get_outputs()]
    active = session.get_providers()
    if active[:1] != providers[:1]:
        # e.g. onnxruntime-gpu with no visible GPU: ONNX Runtime drops CUDA at session start
        logger.warning("Wanted %s first, running on %s.", providers[0], active)
    logger.info("Model loaded with providers %s", active)
    return OnnxModel(session, inputs, outputs, list(inference_cfg.providers))
