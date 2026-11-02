import io
import os

import pytest

BASE_URL = os.getenv("BASE_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")

requests = pytest.importorskip("requests")
Image = pytest.importorskip("PIL.Image")


def _jpeg(size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 140, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def _png(size=(320, 240)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 140, 200)).save(buf, format="PNG")
    return buf.getvalue()


def test_predict_valid_image_returns_schema():
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("sample.jpg", _jpeg(), "image/jpeg")},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image"] == {"width": 320, "height": 240}
    assert body["count"] == len(body["detections"])
    assert isinstance(body["inference_ms"], (int, float))
    for d in body["detections"]:
        assert set(d) == {"class_id", "class", "confidence", "bbox"}
        assert len(d["bbox"]) == 4


def test_predict_non_image_returns_415():
    r = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        timeout=30,
    )
    assert r.status_code == 415


def test_predict_missing_file_returns_422():
    r = requests.post(f"{BASE_URL}/predict", timeout=30)
    assert r.status_code == 422


def test_health_reports_active_providers():
    r = requests.get(f"{BASE_URL}/health", timeout=30)
    assert r.status_code == 200
    io_info = r.json()["model_io"]
    # whatever hardware is present, the service runs on something and ends with CPU as the fallback
    assert io_info["providers"] and io_info["providers"][-1] == "CPUExecutionProvider"
    assert io_info["requested_providers"]


def test_annotate_returns_image():
    r = requests.post(
        f"{BASE_URL}/annotate",
        files={"file": ("sample.png", _png(), "image/png")},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert int(r.headers["x-detections-count"]) >= 0
    assert Image.open(io.BytesIO(r.content)).size == (320, 240)


def test_annotate_non_image_returns_415():
    r = requests.post(
        f"{BASE_URL}/annotate",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
        timeout=30,
    )
    assert r.status_code == 415
