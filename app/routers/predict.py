import logging
import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.pipeline import PipelineError
from app.schemas import ErrorResponse, NotReadyResponse, PredictResponse
from app.utils import (
    ALLOWED_CONTENT_TYPES,
    decode_image,
    read_upload_limited,
    validate_content_type,
    validate_image_dimensions,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


# Plain `def`: FastAPI runs it in a threadpool, so the blocking pipeline
# never stalls the event loop.
@router.post(
    "/predict",
    summary="Run the model on an uploaded image",
    description=f"Upload one image as `file` ({', '.join(ALLOWED_CONTENT_TYPES)}). "
    "Returns detections with boxes as [x1, y1, x2, y2] in original-image pixels.",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Empty, corrupt, or too small image"},
        413: {"model": ErrorResponse, "description": "File larger than inference.max_upload_mb"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "Pipeline or internal error"},
        503: {"model": NotReadyResponse, "description": "Model not loaded. See setup_errors."},
    },
)
def predict(request: Request, file: UploadFile = File(...)):
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
        start = time.perf_counter()
        detections = s.pipeline.run(image)
        inference_ms = (time.perf_counter() - start) * 1000
        return PredictResponse(
            filename=file.filename,
            image={"width": image.width, "height": image.height},
            count=len(detections),
            detections=detections,
            inference_ms=round(inference_ms, 1),
        )
    except PipelineError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Unexpected error in /predict")
        raise HTTPException(status_code=500, detail="Internal server error") from e
