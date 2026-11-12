"""Nhật ký bữa ăn (UC-B04, UC-B05) + thu thập chỉnh sửa khi người ăn đồng ý (BE-F5)."""
from __future__ import annotations

import logging
import shutil
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query, Response
from sqlalchemy import select

from ..db import Correction, Meal, MealItem
from ..deps import CtxDep, DbDep, DeviceDep, live_scan
from ..errors import ApiError
from ..schemas import DayMeals, Error, MealCreate, MealCreated
from ..services import calc

router = APIRouter(prefix="/meals", tags=["meals"])
log = logging.getLogger("calo_khay")


def _to_utc(dt: datetime, tz: ZoneInfo) -> datetime:
    if dt.tzinfo is None:  # không có múi giờ -> coi là giờ Việt Nam
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


@router.post("", response_model=MealCreated, status_code=201, responses={400: {"model": Error}, 422: {"model": Error}},
             summary="Lưu bữa ăn vào nhật ký")
def create_meal(body: MealCreate, ctx: CtxDep, db: DbDep, device: DeviceDep):
    s, tz = ctx.settings, ZoneInfo(ctx.settings.timezone)
    sc = live_scan(db, body.scan_id)  # hết hạn vẫn cho lưu bữa ăn, chỉ không góp ảnh được
    items = calc.recalc(ctx.catalog, body.items, sc.result_json.get("rice_volume") if sc else None)
    tot = calc.total(items, s.daily_kcal, s.error_pct)

    meal = Meal(device_id=device, scan_id=sc.scan_id if sc else None, eaten_at=_to_utc(body.eaten_at, tz),
                total_g=tot["grams"], total_kcal=tot["kcal"],
                model_version=sc.model_version if sc else None,
                items=[MealItem(region_id=i["region_id"], cls=i["cls"], grams=i["grams"], kcal=i["kcal"],
                                source=i["source"], orig_cls=i["orig_cls"], conf=i["conf"], confirmed=i["confirmed"])
                       for i in items])
    db.add(meal)
    db.flush()

    saved = False
    if body.share_corrections and sc is not None and sc.image_path and Path(sc.image_path).exists():
        s.corrections_dir.mkdir(parents=True, exist_ok=True)
        dst = s.corrections_dir / f"meal_{meal.meal_id}.jpg"
        shutil.copyfile(sc.image_path, dst)
        db.add(Correction(meal_id=meal.meal_id, image_path=str(dst), original_json=sc.result_json, corrected_json=items))
        saved = True
    db.commit()
    log.info("meal %s items=%d kcal=%d share=%s", meal.meal_id, len(items), tot["kcal"], saved)
    return dict(meal_id=meal.meal_id, total=tot, correction_saved=saved)


@router.get("", response_model=DayMeals, responses={400: {"model": Error}}, summary="Bữa ăn trong một ngày + tổng ngày")
def list_meals(ctx: CtxDep, db: DbDep, device: DeviceDep,
               day: Annotated[date, Query(alias="date", description="YYYY-MM-DD theo giờ Việt Nam")]):
    tz = ZoneInfo(ctx.settings.timezone)
    start = _to_utc(datetime.combine(day, time.min, tz), tz)
    meals = db.scalars(select(Meal).where(Meal.device_id == device, Meal.eaten_at >= start,
                                          Meal.eaten_at < start + timedelta(days=1)).order_by(Meal.eaten_at)).all()
    out = []
    for m in meals:
        out.append(dict(
            meal_id=m.meal_id, total_g=m.total_g, total_kcal=m.total_kcal, model_version=m.model_version,
            eaten_at=m.eaten_at.replace(tzinfo=timezone.utc).astimezone(tz),
            items=[dict(cls=i.cls, name=ctx.catalog.name(i.cls), grams=i.grams, kcal=i.kcal, source=i.source,
                        orig_cls=i.orig_cls, confirmed=i.confirmed) for i in m.items]))
    k = sum(m.total_kcal for m in meals)
    return dict(date=day, meals=out, total_kcal=k, daily_pct=round(100 * k / ctx.settings.daily_kcal))


@router.delete("/{meal_id}", status_code=204, responses={404: {"model": Error}}, summary="Xoá bữa ăn")
def delete_meal(meal_id: int, db: DbDep, device: DeviceDep):
    m = db.get(Meal, meal_id)
    if m is None or m.device_id != device:
        raise ApiError(404, "meal_not_found")
    img = m.correction.image_path if m.correction else None
    db.delete(m)
    db.commit()
    if img:  # người ăn xoá bữa -> xoá luôn ảnh đã góp
        Path(img).unlink(missing_ok=True)
    return Response(status_code=204)
