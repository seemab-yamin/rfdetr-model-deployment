import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from app.pipeline.base import PipelineError

logger = logging.getLogger(__name__)

PAD_VALUE = 114  # YOLO standard gray


@dataclass(frozen=True)
class ImageSpec:
    name: str
    channels: int
    height: int
    width: int
    layout: str  # "NCHW" or "NHWC"


@dataclass(frozen=True)
class LetterboxContext:
    orig_w: int
    orig_h: int
    scale: float
    pad_x: int
    pad_y: int


def _fixed(dim) -> bool:
    return isinstance(dim, int) and dim > 0


def resolve_image_spec(tensor, configured_size: tuple[int, int]) -> ImageSpec:
    shape = list(tensor.shape)
    if len(shape) != 4:
        raise PipelineError(
            f"The detection pipeline expects a 4D image input (batch, channels, height, width), "
            f"got shape {shape}. Edit app/pipeline/detection/preprocess.py for other inputs."
        )
    nhwc = (
        _fixed(shape[3])
        and shape[3] in (1, 3)
        and not (_fixed(shape[1]) and shape[1] in (1, 3))
    )
    layout = "NHWC" if nhwc else "NCHW"
    c_idx, h_idx, w_idx = (3, 1, 2) if nhwc else (1, 2, 3)

    channels = shape[c_idx] if _fixed(shape[c_idx]) else 3
    height = shape[h_idx] if _fixed(shape[h_idx]) else configured_size[0]
    width = shape[w_idx] if _fixed(shape[w_idx]) else configured_size[1]

    if (
        _fixed(shape[h_idx])
        and _fixed(shape[w_idx])
        and (height, width) != tuple(configured_size)
    ):
        logger.info(
            "Model has a fixed input size %sx%s. Ignoring model.input_size in config.yaml.",
            height,
            width,
        )
    if channels != 3:
        raise PipelineError(
            f"The reference preprocessing supports 3-channel input, model wants {channels}. "
            "Edit app/pipeline/detection/preprocess.py."
        )
    return ImageSpec(tensor.name, channels, height, width, layout)


def letterbox(
    image: Image.Image, height: int, width: int
) -> tuple[np.ndarray, LetterboxContext]:
    """Resize keeping aspect ratio, pad the rest with gray, center the image."""
    image = image.convert("RGB")
    w, h = image.size
    scale = min(width / w, height / h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = np.full((height, width, 3), PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (width - new_w) // 2, (height - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = np.asarray(resized)
    return canvas, LetterboxContext(w, h, scale, pad_x, pad_y)


def to_tensor(
    canvas: np.ndarray, layout: str, channel_order: str, normalize: bool
) -> np.ndarray:
    arr = canvas[..., ::-1] if channel_order == "BGR" else canvas
    arr = arr.astype(np.float32)
    if normalize:
        arr = arr / 255.0
    if layout == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr)[None]
