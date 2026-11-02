from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.config import ConfigError, load_config
from app.pipeline import PipelineError, build_pipeline
from app.routers import health, predict
from app.services.onnx_runtime import ModelLoadError, load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = None
    app.state.model = None
    app.state.pipeline = None
    app.state.model_loaded = False
    app.state.setup_errors = []

    try:
        app.state.settings = load_config()
    except ConfigError as e:
        app.state.setup_errors.append(str(e))

    settings = app.state.settings
    if settings and not settings.model.path.exists():
        app.state.setup_errors.append(
            f"Model file not found: {settings.model.path}. "
            "Put your .onnx file there or update model.path in config.yaml."
        )

    if settings and not app.state.setup_errors:
        try:
            model = load_model(settings.model, settings.inference)
            # lets /health show model_io even if the pipeline fails
            app.state.model = model
            app.state.pipeline = build_pipeline(
                settings.model, settings.inference, model
            )
            app.state.model_loaded = True
        except (ModelLoadError, PipelineError) as e:
            app.state.setup_errors.append(str(e))

    yield

    app.state.model = None
    app.state.pipeline = None
    app.state.model_loaded = False


app = FastAPI(title="Model Service API", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")
