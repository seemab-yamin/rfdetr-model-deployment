from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PipelineError(Exception):
    """Pipeline problem. The message is written for the person setting up the repo."""


class Pipeline(ABC):
    """One task type. Subclass this to add NLP, tabular, or any other task."""

    def __init__(self, model_cfg, inference_cfg, model):
        self.model_cfg = model_cfg
        self.inference_cfg = inference_cfg
        self.model = model

    @abstractmethod
    def preprocess(self, payload: Any) -> tuple[dict[str, np.ndarray], Any]:
        """Return (feed dict for the model, context needed by postprocess)."""

    @abstractmethod
    def postprocess(self, outputs: list[np.ndarray], context: Any) -> Any:
        """Turn raw model outputs into a JSON-safe result."""

    def run(self, payload: Any) -> Any:
        feed, context = self.preprocess(payload)
        outputs = self.model.run(feed)
        return self.postprocess(outputs, context)
