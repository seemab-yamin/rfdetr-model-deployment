import pytest

from app.config import ConfigError, load_config


def write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return p


def test_defaults_when_file_empty(tmp_path):
    s = load_config(write(tmp_path, ""))
    assert s.model.task == "detection"


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_rejects_non_onnx(tmp_path):
    with pytest.raises(ConfigError, match="onnx"):
        load_config(write(tmp_path, "model:\n  path: models/m.pt\n"))


def test_rejects_unknown_key(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "modle:\n  path: x.onnx\n"))


def test_rejects_bad_threshold(tmp_path):
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(write(tmp_path, "inference:\n  confidence_threshold: 2\n"))


def test_model_url_optional(tmp_path):
    assert load_config(write(tmp_path, "")).model.url is None
    s = load_config(write(tmp_path, "model:\n  url: https://example.com/m.onnx\n"))
    assert str(s.model.url) == "https://example.com/m.onnx"


def test_rejects_bad_model_url(tmp_path):
    with pytest.raises(ConfigError, match="model.url"):
        load_config(write(tmp_path, "model:\n  url: not-a-url\n"))


def test_rejects_removed_iou_threshold(tmp_path):
    with pytest.raises(ConfigError, match="iou_threshold"):
        load_config(write(tmp_path, "inference:\n  iou_threshold: 0.45\n"))


def test_providers_env_override(tmp_path, monkeypatch):
    p = write(tmp_path, "inference:\n  providers: [CUDAExecutionProvider]\n  max_upload_mb: 3\n")
    monkeypatch.setenv("INFERENCE_PROVIDERS", "CPUExecutionProvider")
    s = load_config(p)
    assert s.inference.providers == ["CPUExecutionProvider"]
    assert s.inference.max_upload_mb == 3  # other keys kept

    monkeypatch.setenv("INFERENCE_PROVIDERS", " CUDAExecutionProvider , CPUExecutionProvider ")
    assert load_config(write(tmp_path, "")).inference.providers == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
