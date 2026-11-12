from __future__ import annotations

import asyncio
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from backend.db import Scan, make_engine, make_sessionmaker, utcnow
from backend.errors import ApiError
from backend.export_corrections import export
from backend.services import cleanup
from backend.services.catalog import Catalog
from backend.services.inference import Inference

from .conftest import DEVICE, jpeg

H = {"X-Device-Id": DEVICE}


# ------------------------------------------------------------------ info
def test_health_classes_benchmark(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["engine"] == "mock"
    cls = client.get("/api/classes").json()
    assert len(cls) == 45
    drink = next(c for c in cls if c["cls"] == "drink")
    assert drink["kcal_per_100g"] == 0
    rice = next(c for c in cls if c["cls"] == "rice")
    assert rice["default_g"] == 228
    fb = next(c for c in cls if c["cls"] == "fish_ball")
    assert fb["name"] == "cá viên"
    assert client.get("/api/benchmark").status_code in (200, 404)


# ------------------------------------------------------------------ scan
def test_scan_ok(scanned):
    s = scanned
    assert len(s["scan_id"]) == 36 and s["model_version"] == "mock"
    assert s["image"] == {"w": 1000, "h": 800}
    assert [i["cls"] for i in s["items"]] == ["rice", "fish_ball"]
    assert s["items"][1]["needs_confirm"] is True
    assert {"decode", "total"} <= set(s["timing_ms"])


def test_scan_scales_polygons_to_uploaded_image(client):
    r = client.post("/api/scan", files={"image": ("a.png", jpeg(500, 400, fmt="PNG"), "image/png")}).json()
    assert r["image"] == {"w": 500, "h": 400}
    assert r["items"][0]["polygons"][0][1] == [200, 50]


def test_scan_flag_conf_param(client):
    r = client.post("/api/scan?flag_conf=0.3", files={"image": ("a.jpg", jpeg(), "image/jpeg")}).json()
    assert r["items"][1]["needs_confirm"] is False


@pytest.mark.parametrize("data,code,status", [
    (b"", "invalid_image", 400),
    (b"%PDF-1.4 fake pdf", "invalid_image", 400),
    (None, "image_too_small", 400),       # ảnh 1x1
    ("gif", "invalid_image", 400),
])
def test_scan_bad_input(client, data, code, status):
    if data is None:
        data = jpeg(1, 1)
    elif data == "gif":
        data = jpeg(400, 400, fmt="GIF")
    r = client.post("/api/scan", files={"image": ("anh.jpg", data, "image/jpeg")})  # đuôi .jpg nhưng nội dung khác
    assert r.status_code == status and r.json()["error"] == code


def test_scan_too_large(client):
    big = b"\xff\xd8" + b"0" * (10 * 1024 * 1024 + 10)
    r = client.post("/api/scan", files={"image": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413 and r.json()["error"] == "image_too_large"


def test_scan_blurry_warning(client):
    r = client.post("/api/scan", files={"image": ("a.jpg", jpeg(sharp=False), "image/jpeg")}).json()
    assert "blurry" in r["warnings"]


def test_scan_depth_unavailable(client):
    r = client.post("/api/scan?depth=true", files={"image": ("a.jpg", jpeg(), "image/jpeg")}).json()
    assert "depth_unavailable" in r["warnings"]


def test_scan_deterministic(client):
    a = client.post("/api/scan", files={"image": ("a.jpg", jpeg(), "image/jpeg")}).json()
    b = client.post("/api/scan", files={"image": ("a.jpg", jpeg(), "image/jpeg")}).json()
    for k in ("items", "total", "warnings"):
        assert a[k] == b[k]


def test_busy_and_loading(settings):
    inf = Inference(settings)
    img = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ApiError) as e:
        asyncio.run(inf.analyze(img, None, None))
    assert e.value.code == "model_loading"
    inf.status, inf.pending = "ok", settings.max_queue + 1
    with pytest.raises(ApiError) as e:
        asyncio.run(inf.analyze(img, None, None))
    assert e.value.code == "busy" and e.value.status == 503


# ------------------------------------------------------------------ recalc
def test_recalc_rules(client, scanned):
    cat = Catalog(Path(__file__).resolve().parents[2] / "models")
    items = [
        {"region_id": 1, "cls": "rice", "source": "model", "area_ratio": 0.3, "grams": 150},        # chọn suất cơm
        {"region_id": 2, "cls": "braised_egg", "source": "renamed", "orig_cls": "fish_ball", "area_ratio": 0.075},
        {"cls": "drink", "source": "manual"},
        {"cls": "potato", "source": "manual", "grams": 120},
    ]
    r = client.post("/api/recalc", json={"scan_id": scanned["scan_id"], "items": items})
    assert r.status_code == 200, r.text
    out = r.json()
    rice, egg, drink, potato = out["items"]
    assert (rice["grams"], rice["gram_method"]) == (150, "user")
    assert egg["grams"] == cat.grams("braised_egg", 0.075)[0] and egg["gram_method"] == "area"
    assert egg["kcal"] == round(egg["grams"] * cat.rows["braised_egg"]["kcal_per_100g"] / 100)
    assert drink["kcal"] == 0 and drink["grams"] == cat.default_g("drink")
    assert potato["grams"] == 120 and potato["region_id"] is None
    t = out["total"]
    assert t["kcal"] == sum(i["kcal"] for i in out["items"])
    assert t["daily_pct"] == round(100 * t["kcal"] / 2000) and t["error_pct"] == 30


def test_recalc_errors(client):
    r = client.post("/api/recalc", json={"items": [{"cls": "pizza", "source": "manual"}]})
    assert r.status_code == 422 and r.json()["error"] == "unknown_class"
    r = client.post("/api/recalc", json={"scan_id": "khong-ton-tai", "items": []})
    assert r.status_code == 404 and r.json()["error"] == "scan_expired"
    r = client.post("/api/recalc", json={"items": [{"cls": "rice", "source": "abc"}]})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


# ------------------------------------------------------------------ meals
def _meal(scan_id, eaten_at="2026-11-12T12:30:00+07:00", share=False):
    return {"scan_id": scan_id, "eaten_at": eaten_at, "share_corrections": share, "items": [
        {"region_id": 1, "cls": "rice", "source": "model", "area_ratio": 0.3, "grams": 227},
        {"region_id": 2, "cls": "fish_ball", "source": "model", "area_ratio": 0.075, "conf": 0.42, "confirmed": False},
    ]}


def test_meal_flow(client, scanned):
    assert client.post("/api/meals", json=_meal(scanned["scan_id"])).json()["error"] == "missing_device_id"
    r = client.post("/api/meals", json=_meal(scanned["scan_id"]), headers=H)
    assert r.status_code == 201, r.text
    meal = r.json()
    assert meal["correction_saved"] is False

    day = client.get("/api/meals?date=2026-11-12", headers=H).json()
    assert len(day["meals"]) == 1 and day["total_kcal"] == meal["total"]["kcal"]
    m = day["meals"][0]
    assert m["eaten_at"].startswith("2026-11-12T12:30:00") and m["eaten_at"].endswith("+07:00")
    assert m["items"][1]["confirmed"] is False and m["items"][0]["name"] == "cơm trắng"

    other = {"X-Device-Id": "another-device-01"}
    assert client.get("/api/meals?date=2026-11-12", headers=other).json()["meals"] == []
    assert client.delete(f"/api/meals/{meal['meal_id']}", headers=other).status_code == 404
    assert client.delete(f"/api/meals/{meal['meal_id']}", headers=H).status_code == 204
    assert client.get("/api/meals?date=2026-11-12", headers=H).json()["meals"] == []


def test_meal_day_boundary_vietnam_time(client, scanned):
    client.post("/api/meals", json=_meal(scanned["scan_id"], "2026-11-12T23:30:00+07:00"), headers=H)
    client.post("/api/meals", json=_meal(scanned["scan_id"], "2026-11-13T00:30:00+07:00"), headers=H)
    client.post("/api/meals", json=_meal(scanned["scan_id"], "2026-11-12T18:00:00"), headers=H)  # không múi giờ = giờ VN
    assert len(client.get("/api/meals?date=2026-11-12", headers=H).json()["meals"]) == 2
    assert len(client.get("/api/meals?date=2026-11-13", headers=H).json()["meals"]) == 1


def test_meal_without_scan_and_expired_scan(client):
    body = _meal(None)
    body["items"] = [{"cls": "potato", "source": "manual"}]
    r = client.post("/api/meals", json=body, headers=H)
    assert r.status_code == 201
    body["scan_id"], body["share_corrections"] = "het-han", True
    r = client.post("/api/meals", json=body, headers=H)
    assert r.status_code == 201 and r.json()["correction_saved"] is False


def test_share_corrections_and_export(client, scanned, settings, tmp_path):
    body = _meal(scanned["scan_id"], share=True)
    body["items"][1].update(cls="braised_egg", source="renamed", orig_cls="fish_ball")
    body["items"].append({"cls": "potato", "source": "manual"})
    r = client.post("/api/meals", json=body, headers=H).json()
    assert r["correction_saved"] is True
    img = settings.corrections_dir / f"meal_{r['meal_id']}.jpg"
    assert img.exists()

    out = tmp_path / "export"
    assert export(out, settings) == 1
    labels = (out / "labels" / img.with_suffix(".txt").name).read_text().splitlines()
    cat = Catalog(settings.models_dir)
    assert {int(l.split()[0]) for l in labels} == {cat.rows["rice"]["id"], cat.rows["braised_egg"]["id"]}
    assert all(0 <= float(v) <= 1 for l in labels for v in l.split()[1:])
    assert "potato" in (out / "missing.csv").read_text(encoding="utf-8")

    assert client.delete(f"/api/meals/{r['meal_id']}", headers=H).status_code == 204
    assert not img.exists()  # xoá bữa -> xoá ảnh đã góp


# ------------------------------------------------------------------ dọn dẹp 24 giờ
def test_cleanup_purges_expired(client, scanned, settings):
    sessions = make_sessionmaker(make_engine(settings.database_url))
    with sessions() as db:
        sc = db.get(Scan, scanned["scan_id"])
        path = Path(sc.image_path)
        assert path.exists()
        sc.expires_at = utcnow() - timedelta(minutes=1)
        db.commit()
    assert cleanup.purge(sessions, settings.tmp_dir, settings.scan_ttl_hours) == 1
    assert not path.exists()
    r = client.post("/api/recalc", json={"scan_id": scanned["scan_id"], "items": []})
    assert r.json()["error"] == "scan_expired"


def test_catalog_matches_pipeline_formula():
    from calo_khay.pipeline import _loglog
    cat = Catalog(Path(__file__).resolve().parents[2] / "models")
    g, how = cat.grams("potato", 0.15)
    assert how == "area" and g == round(_loglog(cat.gm["area"], "potato", 0.15))
    assert cat.grams("potato", None) == (round(cat.gm["class_mean_g"]["potato"]), "class_mean")
    assert cat.grams("rice", 0.3)[1] == "rice_mean"
    assert not math.isnan(cat.rows["drink"]["kcal_per_100g"])
