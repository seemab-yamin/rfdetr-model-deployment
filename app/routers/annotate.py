import io
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.pipeline import PipelineError
from app.schemas import ErrorResponse, NotReadyResponse
from app.utils import (
    ALLOWED_CONTENT_TYPES,
    decode_image,
    draw_detections,
    read_upload_limited,
    validate_content_type,
    validate_image_dimensions,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Annotation"])

# Pillow format name -> response content type
FORMAT_CONTENT_TYPES = {fmt: ct for ct, fmt in ALLOWED_CONTENT_TYPES.items()}


@router.post(
    "/annotate",
    summary="Draw detections on an uploaded image",
    description=f"Upload one image as `file` ({', '.join(ALLOWED_CONTENT_TYPES)}). "
    "Returns the same image, in the same format, with boxes and labels drawn on it.",
    response_class=StreamingResponse,
    responses={
        200: {"content": {ct: {} for ct in ALLOWED_CONTENT_TYPES}, "description": "Annotated image"},
        400: {"model": ErrorResponse, "description": "Empty, corrupt, or too small image"},
        413: {"model": ErrorResponse, "description": "File larger than inference.max_upload_mb"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "Pipeline or internal error"},
        503: {"model": NotReadyResponse, "description": "Model not loaded. See setup_errors."},
    },
)
def annotate(request: Request, file: UploadFile = File(...)):
    s = request.app.state
    if s.setup_errors or not s.model_loaded:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service not ready", "setup_errors": s.setup_errors},
        )

    validate_content_type(file)
    raw = read_upload_limited(file, s.settings.inference.max_upload_mb)
    image = decode_image(raw)
    validate_image_dimensions(image)

    try:
        detections = s.pipeline.run(image)
        annotated = draw_detections(image, detections)
    except PipelineError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error in /annotate")
        raise HTTPException(status_code=500, detail="Internal server error") from e

    buffer = io.BytesIO()
    annotated.save(buffer, format=image.format)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type=FORMAT_CONTENT_TYPES[image.format],
        headers={"X-Detections-Count": str(len(detections))},
    )
