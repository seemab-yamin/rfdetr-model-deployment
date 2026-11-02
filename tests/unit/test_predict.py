import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipeline import PipelineError

TWO_DETECTIONS = [
    {"class_id": 5, "class": "bus", "confidence": 0.91, "bbox": [22.0, 231.5, 805.0, 756.8]},
    {"class_id": 0, "class": "person", "confidence": 0.87, "bbox": [48.0, 398.0, 245.0, 902.0]},
]


class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else []
        self.error = error

    def run(self, image):
        if self.error:
            raise self.error
        return self.result


def _make_project(tmp_path, max_upload_mb=None):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.onnx").write_bytes(b"")
    config = "model:\n  path: models/model.onnx\n"
    if max_upload_mb is not None:
        config += f"inference:\n  max_upload_mb: {max_upload_mb}\n"
    (tmp_path / "config.yaml").write_text(config)


@pytest.fixture
def ready_app(tmp_path, monkeypatch):
    """Start the app with a fake model and yield a function that opens a client."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "app.main.load_model", lambda m, i: SimpleNamespace(info=lambda: {})
    )
    monkeypatch.setattr("app.main.build_pipeline", lambda m, i, model: object())
    return tmp_path


def image_bytes(fmt="JPEG", size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 30, 200)).save(buf, format=fmt)
    return buf.getvalue()


def post_image(client, data, content_type="image/jpeg", filename="bus.jpg"):
    return client.post("/predict", files={"file": (filename, data, content_type)})


def test_success_returns_full_schema(ready_app):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline(TWO_DETECTIONS)
        r = post_image(c, image_bytes(size=(81, 108)))

    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"filename", "image", "count", "detections", "inference_ms"}
    assert body["filename"] == "bus.jpg"
    assert body["image"] == {"width": 81, "height": 108}
    assert body["count"] == 2
    assert body["detections"] == TWO_DETECTIONS
    assert isinstance(body["inference_ms"], float) and body["inference_ms"] >= 0


def test_empty_detections_is_not_an_error(ready_app):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline([])
        r = post_image(c, image_bytes())

    assert r.status_code == 200
    assert r.json()["count"] == 0 and r.json()["detections"] == []


@pytest.mark.parametrize(
    "fmt, content_type",
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_accepts_jpeg_png_webp(ready_app, fmt, content_type):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline(TWO_DETECTIONS[:1])
        r = post_image(c, image_bytes(fmt), content_type, f"img.{fmt.lower()}")

    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_503_lists_setup_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no config.yaml, no model
    with TestClient(app) as c:
        r = post_image(c, image_bytes())

    assert r.status_code == 503
    body = r.json()
    assert body["detail"] == "Service not ready"
    assert body["setup_errors"] and any("config" in e.lower() for e in body["setup_errors"])


def test_503_when_model_not_loaded(ready_app):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.model_loaded = False
        r = post_image(c, image_bytes())

    assert r.status_code == 503
    assert r.json() == {"detail": "Service not ready", "setup_errors": []}


@pytest.mark.parametrize(
    "data, content_type",
    [
        (b"hello, not an image", "text/plain"),
        (image_bytes(), "application/octet-stream"),
        (image_bytes("GIF"), "image/gif"),
    ],
)
def test_415_wrong_content_type(ready_app, data, content_type):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline()
        r = post_image(c, data, content_type, "file.bin")

    assert r.status_code == 415
    assert "image/jpeg" in r.json()["detail"]
    assert "image/png" in r.json()["detail"] and "image/webp" in r.json()["detail"]


def test_415_when_content_does_not_match_allowed_types(ready_app):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline()
        r = post_image(c, image_bytes("GIF"), "image/png", "fake.png")

    assert r.status_code == 415


def test_413_file_above_limit(ready_app):
    _make_project(ready_app, max_upload_mb=1)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline()
        r = post_image(c, b"\xff" * (1024 * 1024 + 1))

    assert r.status_code == 413
    assert "1 MB" in r.json()["detail"]


@pytest.mark.parametrize(
    "data, expected",
    [
        (b"", "empty"),
        (b"\xff\xd8\xff\xe0 definitely not a jpeg", "corrupt"),
        (image_bytes()[:200], "corrupt"),  # truncated, caught by .load()
        (image_bytes(size=(4, 4)), "too small"),
    ],
    ids=["empty", "garbage", "truncated", "4x4"],
)
def test_400_bad_image(ready_app, data, expected):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline()
        r = post_image(c, data)

    assert r.status_code == 400
    assert expected in r.json()["detail"].lower()


def test_422_when_file_missing(ready_app):
    _make_project(ready_app)
    with TestClient(app) as c:
        r = c.post("/predict")

    assert r.status_code == 422


def test_pipeline_error_returns_its_message(ready_app):
    _make_project(ready_app)
    msg = "config.yaml model.labels has 2 names but the model outputs 80 classes."
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline(error=PipelineError(msg))
        r = post_image(c, image_bytes())

    assert r.status_code == 500
    assert r.json() == {"detail": msg}


def test_unexpected_error_is_generic_500(ready_app, caplog):
    _make_project(ready_app)
    with TestClient(app) as c:
        c.app.state.pipeline = FakePipeline(error=RuntimeError("secret internals"))
        r = post_image(c, image_bytes())

    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert "secret internals" not in r.text and "Traceback" not in r.text
    assert "secret internals" in caplog.text  # logged with traceback instead


def test_openapi_documents_error_responses():
    responses = app.openapi()["paths"]["/predict"]["post"]["responses"]
    assert {"200", "400", "413", "415", "422", "500", "503"} <= set(responses)
