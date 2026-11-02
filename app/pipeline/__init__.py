from app.pipeline.base import Pipeline, PipelineError
from app.pipeline.detection import DetectionPipeline

# To add a task: write a Pipeline subclass, register it here,
# and add its name to the `task` Literal in app/config.py.
PIPELINES: dict[str, type[Pipeline]] = {
    "detection": DetectionPipeline,
}


def build_pipeline(model_cfg, inference_cfg, model) -> Pipeline:
    return PIPELINES[model_cfg.task](model_cfg, inference_cfg, model)
