import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
)

CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "config.yaml"))
# Comma-separated override for inference.providers, e.g. CPUExecutionProvider for local CPU dev
PROVIDERS_ENV = "INFERENCE_PROVIDERS"


class ConfigError(Exception):
    """Config problem. The message is written for the person setting up the repo."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos fail loudly


class ModelConfig(_Section):
    path: Path = Path("models/model.onnx")
    url: HttpUrl | None = None  # CI downloads the model from here to `path`
    task: Literal["detection"] = "detection"
    input_size: tuple[int, int] = (640, 640)
    channel_order: Literal["RGB", "BGR"] = "RGB"
    normalize: bool = True
    mean: tuple[float, float, float] | None = (
        None  # per-channel, applied after normalize
    )
    std: tuple[float, float, float] | None = None
    labels: list[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def must_be_onnx(cls, v: Path) -> Path:
        if v.suffix.lower() != ".onnx":
            raise ValueError("must point to a .onnx file")
        return v


class InferenceConfig(_Section):
    providers: list[str] = ["CPUExecutionProvider"]
    confidence_threshold: float = Field(0.25, ge=0, le=1)
    max_upload_mb: int = Field(10, gt=0)


class Settings(_Section):
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()


def load_config(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. Create it in the project root. "
            "See docs/first-time-setup.md."
        )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path} is not valid YAML: {e}") from e
    if isinstance(raw, dict) and (env := os.getenv(PROVIDERS_ENV, "").strip()):
        providers = [p.strip() for p in env.split(",") if p.strip()]
        raw["inference"] = {**(raw.get("inference") or {}), "providers": providers}
    try:
        return Settings(**raw)
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()
        )
        raise ConfigError(f"Invalid {path}: {problems}") from e
