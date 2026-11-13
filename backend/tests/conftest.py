"""Test chạy bằng SQLite + engine mock: không cần MySQL, GPU hay file model .pt."""
from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app
from backend.config import ROOT, Settings

DEVICE = "test-device-0001"

MOCK = {
    "image": {"w": 1000, "h": 800}, "plate_found": True, "plate_px": 400000, "plate_ellipse": False,
    "rice_volume": None,
    "items": [
        {"id": 1, "cls": "rice", "name": "cơm trắng", "conf": 0.95, "needs_confirm": False, "suggestions": [],
         "area_px": 120000, "area_ratio": 0.3, "grams": 228, "gram_method": "rice_mean", "kcal": 296,
         "polygons": [[[100, 100], [400, 100], [400, 400], [100, 400]]]},
        {"id": 2, "cls": "fish_ball", "name": "cá viên", "conf": 0.42, "needs_confirm": True,
         "suggestions": ["braised_egg", "fried_egg"], "area_px": 30000, "area_ratio": 0.075, "grams": 60,
         "gram_method": "area", "kcal": 100, "polygons": [[[500, 500], [600, 500], [600, 600]]]},
    ],
    "total": {"grams": 288, "kcal": 396, "error_pct": 30, "daily_pct": 20},
    "warnings": [], "timing_ms": {"segment": 1},
}


def jpeg(w=1000, h=800, sharp=True, fmt="JPEG") -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8) if sharp else np.full((h, w, 3), 128, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    mock = tmp_path / "scan.json"
    mock.write_text(json.dumps(MOCK, ensure_ascii=False), encoding="utf-8")
    return Settings(_env_file=None, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}", engine="mock",
                    mock_scan=mock, data_dir=tmp_path / "data", models_dir=ROOT / "models", cleanup_interval_s=3600)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        for _ in range(100):  # model nạp ở luồng nền
            if c.get("/api/health").status_code == 200:
                break
            time.sleep(0.02)
        yield c


@pytest.fixture
def scanned(client):
    r = client.post("/api/scan", files={"image": ("khay.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    return r.json()
