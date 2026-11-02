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
