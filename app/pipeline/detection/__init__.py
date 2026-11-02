from app.pipeline.base import Pipeline, PipelineError
from app.pipeline.detection.postprocess import postprocess
from app.pipeline.detection.preprocess import resize, resolve_image_spec, to_tensor


class DetectionPipeline(Pipeline):
    def __init__(self, model_cfg, inference_cfg, model):
        super().__init__(model_cfg, inference_cfg, model)

        if len(model.inputs) != 1:
            names = [i.name for i in model.inputs]
            raise PipelineError(
                f"The detection pipeline expects 1 model input, found {len(names)}: {names}. "
                "Edit app/pipeline/detection/."
            )
        self.spec = resolve_image_spec(model.inputs[0], model_cfg.input_size)

    def preprocess(self, image):
        canvas, ctx = resize(image, self.spec.height, self.spec.width)
        tensor = to_tensor(
            canvas,
            self.spec.layout,
            self.model_cfg.channel_order,
            self.model_cfg.normalize,
            self.model_cfg.mean,
            self.model_cfg.std,
        )
        return {self.spec.name: tensor}, ctx

    def postprocess(self, outputs, ctx):
        return postprocess(
            outputs,
            ctx,
            self.model_cfg.labels,
            self.inference_cfg.confidence_threshold,
        )
