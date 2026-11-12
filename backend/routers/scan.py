"""POST /api/scan (UC-B01) và POST /api/recalc (UC-B03)."""
from __future__ import annotations

import logging
import time
import uuid
from datetime import timedelta
from typing import Annotated

import cv2
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from ..db import Scan, utcnow
from ..deps import CtxDep, DbDep, live_scan, optional_device_id
from ..errors import ApiError
from ..schemas import Error, RecalcRequest, RecalcResponse, ScanResponse
from ..services import calc
from ..services.image_check import decode, sharpness

router = APIRouter(tags=["scan"])
log = logging.getLogger("calo_khay")

ERRORS = {400: {"model": Error}, 413: {"model": Error}, 422: {"model": Error}, 503: {"model": Error}}


@router.post("/scan", response_model=ScanResponse, responses=ERRORS,
             summary="Phân tích ảnh khay: món, gram, kcal")
async def scan(
    request: Request,
    ctx: CtxDep,
    image: Annotated[UploadFile, File(description="ảnh JPG/PNG/WebP, tối đa 10 MB, cạnh ngắn >= 320 px")],
    depth: Annotated[bool | None, Query(description="bật/tắt độ sâu cho cơm; bỏ trống = theo cấu hình server")] = None,
    flag_conf: Annotated[float | None, Query(ge=0, le=1, description="ngưỡng 'Cần xác nhận'; bỏ trống = mặc định")] = None,
    device: Annotated[str | None, Depends(optional_device_id)] = None,
):
    s = ctx.settings
    t0 = time.perf_counter()
    max_bytes = int(s.max_image_mb * 1024 * 1024)
    if int(request.headers.get("content-length") or 0) > max_bytes + 64 * 1024:
        raise ApiError(413, "image_too_large")
    data = await image.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ApiError(413, "image_too_large")
    bgr = await run_in_threadpool(decode, data, s.min_image_side)
    blurry = await run_in_threadpool(sharpness, bgr) < s.blur_threshold
    t_decode = time.perf_counter() - t0

    res = await ctx.inference.analyze(bgr, flag_conf, depth)
    if blurry:
        res["warnings"].append("blurry")
    for it in res["items"]:
        it.setdefault("source", "model")

    scan_id = str(uuid.uuid4())
    path = s.tmp_dir / f"{scan_id}.jpg"  # ảnh đã xoay EXIF, khớp toạ độ polygons; tự xoá sau 24 giờ (QT-09)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        await run_in_threadpool(path.write_bytes, buf.tobytes())

    timing = dict(decode=round(t_decode * 1000), **res.pop("timing_ms", {}))
    timing["total"] = round((time.perf_counter() - t0) * 1000)
    out = dict(scan_id=scan_id, model_version=ctx.inference.version, timing_ms=timing, **res)

    def save():
        with ctx.sessions() as db:
            db.add(Scan(scan_id=scan_id, device_id=device, result_json=out, image_path=str(path) if ok else None,
                        model_version=ctx.inference.version, expires_at=utcnow() + timedelta(hours=s.scan_ttl_hours)))
            db.commit()
    await run_in_threadpool(save)

    log.info("scan %s items=%d warnings=%s timing=%s model=%s", scan_id, len(out["items"]),
             ",".join(out["warnings"]) or "-", timing, ctx.inference.version)  # không log ảnh (NFR-B11)
    return out


@router.post("/recalc", response_model=RecalcResponse, responses=ERRORS,
             summary="Tính lại gram/kcal sau khi đổi, thêm, xoá món hoặc chọn suất cơm")
def recalc(body: RecalcRequest, ctx: CtxDep, db: DbDep):
    rice_volume = None
    if body.scan_id:
        sc = live_scan(db, body.scan_id)
        if sc is None:
            raise ApiError(404, "scan_expired")
        rice_volume = sc.result_json.get("rice_volume")
    items = calc.recalc(ctx.catalog, body.items, rice_volume)
    return dict(items=items, total=calc.total(items, ctx.settings.daily_kcal, ctx.settings.error_pct))
