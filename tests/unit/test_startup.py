from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.services.onnx_runtime import ModelLoadError

def _make_project(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.onnx").write_bytes(b"")
    (tmp_path / "config.yaml").write_text("model:\n  path: models/model.onnx\n")


def test_model_load_failure_reported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)

    def boom(m, i):
        raise ModelLoadError("Could not load models/model.onnx as an ONNX model: bad")

    monkeypatch.setattr("app.main.load_model", boom)
    with TestClient(app) as c:
        assert any(
            "Could not load" in e for e in c.get("/health").json()["setup_errors"]
        )
        assert c.get("/ping").status_code == 503


def test_model_loaded_makes_ping_ready(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    fake = SimpleNamespace(info=lambda: {"inputs": []})
    monkeypatch.setattr("app.main.load_model", lambda m, i: fake)
    monkeypatch.setattr("app.main.build_pipeline", lambda m, i, model: object())
    with TestClient(app) as c:
        body = c.get("/health").json()
        assert body["model_loaded"] is True and body["setup_errors"] == []
        assert c.get("/ping").status_code == 200


def test_pipeline_error_reported(tmp_path, monkeypatch):
    from app.pipeline import PipelineError

    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path)
    monkeypatch.setattr(
        "app.main.load_model", lambda m, i: SimpleNamespace(info=lambda: {})
    )

    def boom(m, i, model):
        raise PipelineError("expects a 4D image input")

    monkeypatch.setattr("app.main.build_pipeline", boom)
    with TestClient(app) as c:
        assert any("4D" in e for e in c.get("/health").json()["setup_errors"])
        assert c.get("/ping").status_code == 503
