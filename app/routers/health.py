from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(request: Request):
    s = request.app.state
    return {
        "status": "running",
        "model_loaded": s.model_loaded,
        "model_path": str(s.settings.model.path) if s.settings else None,
        "model_io": s.model.info() if s.model else None,
        "setup_errors": s.setup_errors,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/ping")
def ping(request: Request):
    s = request.app.state
    if s.setup_errors or not s.model_loaded:
        raise HTTPException(status_code=503, detail="Not ready. See /health.")
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
