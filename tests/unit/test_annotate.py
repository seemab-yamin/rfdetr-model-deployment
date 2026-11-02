import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.pipeline import PipelineError
from app.utils import draw_detections

BUS = {"class_id": 6, "class": "bus", "confidence": 0.95, "bbox": [10.0, 10.0, 50.0, 40.0]}


class FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else []
        self.error = error

    def run(self, image):
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.onnx").write_bytes(b"")
    (tmp_path / "config.yaml").write_text("model:\n  path: models/model.onnx\n")
    monkeypatch.setattr("app.main.load_model", lambda m, i: SimpleNamespace(info=lambda: {}))
    monkeypatch.setattr("app.main.build_pipeline", lambda m, i, model: object())
    with TestClient(app) as c:
        yield c


def image_bytes(fmt="JPEG", size=(64, 48), color=(0, 0, 255)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def post(c, data, content_type="image/jpeg", filename="img.jpg"):
    return c.post("/annotate", files={"file": (filename, data, content_type)})


@pytest.mark.parametrize(
    "fmt, content_type",
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_returns_same_format_with_count_header(client, fmt, content_type):
    client.app.state.pipeline = FakePipeline([BUS])
    r = post(client, image_bytes(fmt), content_type, f"img.{fmt.lower()}")

    assert r.status_code == 200
    assert r.headers["content-type"] == content_type
    assert r.headers["x-detections-count"] == "1"
    out = Image.open(io.BytesIO(r.content))
    assert out.format == fmt and out.size == (64, 48)


def test_no_detections_returns_unchanged_image(client):
    client.app.state.pipeline = FakePipeline([])
    r = post(client, image_bytes("PNG"), "image/png", "img.png")

    assert r.status_code == 200 and r.headers["x-detections-count"] == "0"
    assert Image.open(io.BytesIO(r.content)).getpixel((5, 5)) == (0, 0, 255)


def test_503_when_model_not_loaded(client):
    client.app.state.model_loaded = False
    r = post(client, image_bytes())
    assert r.status_code == 503
    assert r.json() == {"detail": "Service not ready", "setup_errors": []}


@pytest.mark.parametrize(
    "data, content_type, status",
    [
        (b"not an image", "text/plain", 415),
        (b"", "image/jpeg", 400),
        (image_bytes(size=(4, 4)), "image/jpeg", 400),
    ],
    ids=["wrong-type", "empty", "too-small"],
)
def test_rejects_bad_uploads(client, data, content_type, status):
    client.app.state.pipeline = FakePipeline()
    assert post(client, data, content_type).status_code == status


def test_pipeline_error_returns_its_message(client):
    client.app.state.pipeline = FakePipeline(error=PipelineError("labels mismatch"))
    r = post(client, image_bytes())
    assert r.status_code == 500 and r.json() == {"detail": "labels mismatch"}


def test_draw_detections_draws_box_and_keeps_original():
    original = Image.new("RGB", (64, 48), (0, 0, 255))
    out = draw_detections(original, [BUS])
    assert out.getpixel((10, 25)) == (255, 0, 0)  # left edge of the box is red
    assert out.getpixel((30, 25)) == (0, 0, 255)  # inside the box untouched
    assert original.getpixel((10, 25)) == (0, 0, 255)  # input image not modified


def test_openapi_documents_annotate():
    responses = app.openapi()["paths"]["/annotate"]["post"]["responses"]
    assert {"200", "400", "413", "415", "422", "500", "503"} <= set(responses)
