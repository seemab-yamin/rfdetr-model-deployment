import io

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# Content types accepted by /predict, and the Pillow format each must decode to
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

MIN_SIDE_PX = 16
READ_CHUNK_BYTES = 1024 * 1024


def validate_content_type(file: UploadFile) -> None:
    """Raise 415 unless the upload declares an allowed image type."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )


def read_upload_limited(file: UploadFile, max_mb: int) -> bytes:
    """
    Read an upload in chunks and stop as soon as it exceeds max_mb (413).
    Never holds more than max_mb plus one chunk in memory.
    Sync on purpose: call it from a plain `def` route.
    """
    max_bytes = max_mb * 1024 * 1024
    buffer = bytearray()
    while chunk := file.file.read(READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max allowed: {max_mb} MB.",
            )
    return bytes(buffer)


def decode_image(image_bytes: bytes) -> Image.Image:
    """
    Decode bytes with Pillow. load() forces a full decode so truncated files fail here.
    Raises 400 for empty or corrupt data, 415 if the real format is not allowed.
    """
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {e}",
        ) from e
    except Image.DecompressionBombError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if image.format not in ALLOWED_CONTENT_TYPES.values():
        raise HTTPException(
            status_code=415,
            detail=f"File content is {image.format}, not an allowed image type. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )
    return image


def validate_image_dimensions(
    image: Image.Image,
    min_side: int = MIN_SIDE_PX,
    max_side: int = 8192,
) -> None:
    """
    Guard against absurdly small or large images before inference.
    """
    width, height = image.size
    if min(width, height) < min_side:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({width}x{height}). Min side: {min_side}px.",
        )
    if max(width, height) > max_side:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({width}x{height}). Max side: {max_side}px.",
        )


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw [x1, y1, x2, y2] boxes and "class confidence" labels on a copy of the image."""
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width = max(2, round(min(image.size) / 300))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=width)
        label = f"{det['class']} {det['confidence']:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        # label above the box, or inside it when the box touches the top edge
        label_y = y1 - text_h - 4 if y1 >= text_h + 4 else y1
        draw.rectangle([x1, label_y, x1 + text_w + 4, label_y + text_h + 4], fill="red")
        draw.text((x1 + 2 - left, label_y + 2 - top), label, fill="white", font=font)

    return image
