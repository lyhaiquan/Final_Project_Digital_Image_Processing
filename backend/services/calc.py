"""Tính lại gram/kcal sau khi người ăn sửa (UC-B03). Không cần ảnh, không dùng GPU."""
from __future__ import annotations

from ..errors import ApiError
from ..schemas import ItemIn
from .catalog import Catalog


def total(items: list[dict], daily_kcal: int, error_pct: int) -> dict:
    g = sum(i["grams"] for i in items)
    k = sum(i["kcal"] for i in items)
    return dict(grams=g, kcal=k, error_pct=error_pct, daily_pct=round(100 * k / daily_kcal))


def recalc(cat: Catalog, items: list[ItemIn], rice_volume: float | None = None) -> list[dict]:
    """Quy tắc:
    - có `grams` (người ăn chọn suất cơm / chỉnh tay) -> lấy đúng con số đó (FR-B14);
    - manual -> khẩu phần trung bình của món (FR-B13);
    - model / renamed -> công thức của món (cuối cùng) trên area_ratio của vùng (FR-B12);
      cơm dùng thể tích V của lần quét nếu có (bật độ sâu).
    """
    out = []
    for it in items:
        if it.cls not in cat:
            raise ApiError(422, "unknown_class", f"Mã món không có trong danh mục: {it.cls}")
        if it.orig_cls is not None and it.orig_cls not in cat:
            raise ApiError(422, "unknown_class", f"Mã món không có trong danh mục: {it.orig_cls}")
        if it.grams is not None:
            g, how = it.grams, "user"
        elif it.source == "manual":
            g, how = cat.default_g(it.cls), ("rice_mean" if it.cls == "rice" else "class_mean")
        else:
            g, how = cat.grams(it.cls, it.area_ratio, rice_volume)
        d = it.model_dump()
        d.update(name=cat.name(it.cls), grams=g, kcal=cat.kcal(it.cls, g), gram_method=how)
        if it.source == "manual":
            d["region_id"] = None
        out.append(d)
    return out
