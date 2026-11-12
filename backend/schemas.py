"""Hợp đồng JSON giữa backend và frontend (mục 2.5 SRS). Mỗi trường có mô tả để /docs tự đầy đủ."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["model", "renamed", "manual"]


class Error(BaseModel):
    error: str = Field(description="mã lỗi, vd. image_too_small (bảng 2.6)")
    message: str = Field(description="câu thông báo tiếng Việt, frontend có thể hiện thẳng")


# ------------------------------------------------------------------ chung
class Total(BaseModel):
    grams: int = Field(description="tổng gram cả khay")
    kcal: int = Field(description="tổng kcal cả khay")
    error_pct: int = Field(description="sai số hiển thị cho mỗi món, %, (QT-06)")
    daily_pct: int = Field(description="% so với nhu cầu 2.000 kcal/ngày (QT-11)")


# ------------------------------------------------------------------ /api/scan
class ScanItem(BaseModel):
    id: int = Field(description="số thứ tự vùng món trong lần quét, dùng làm region_id khi sửa")
    cls: str = Field(description="mã món tiếng Anh, vd. fish_ball")
    name: str = Field(description="tên tiếng Việt")
    conf: float = Field(description="độ tin cậy 0-1")
    needs_confirm: bool = Field(description="true -> giao diện hỏi lại người ăn")
    suggestions: list[str] = Field(description="mã các món có thể đúng hơn, tối đa 3")
    area_px: int = Field(description="diện tích món (pixel)")
    area_ratio: float | None = Field(description="diện tích món / diện tích đĩa; null khi không thấy đĩa")
    grams: int = Field(description="gram ước lượng")
    gram_method: str = Field(description="area | rice_mean | rice_depth | class_mean")
    kcal: int = Field(description="kcal ước lượng")
    polygons: list[list[list[int]]] = Field(description="viền món [[[x, y], ...], ...] theo pixel ảnh đã xoay EXIF")


class ImageSize(BaseModel):
    w: int
    h: int


class ScanResponse(BaseModel):
    scan_id: str = Field(description="mã lần quét; gửi lại cho /recalc và /meals; hết hạn sau 24 giờ")
    image: ImageSize = Field(description="kích thước ảnh đã xoay EXIF, là hệ toạ độ của polygons")
    plate_found: bool
    items: list[ScanItem] = Field(description="mỗi món một dòng, sắp theo diện tích giảm dần")
    total: Total
    warnings: list[str] = Field(description="no_food, no_plate, depth_plane_failed, depth_unavailable, blurry")
    model_version: str
    timing_ms: dict[str, int] = Field(description="thời gian từng bước (ms): decode, segment, plate, depth, grams, total")


# ------------------------------------------------------------------ /api/classes
class FoodClass(BaseModel):
    cls: str = Field(description="mã món")
    name: str = Field(description="tên tiếng Việt")
    kcal_per_100g: float
    default_g: int = Field(description="khẩu phần trung bình, dùng khi thêm tay")
    source: str | None = Field(description="nguồn số liệu kcal")
    suggestions: list[str] = Field(description="các món hay bị nhầm với món này")


# ------------------------------------------------------------------ /api/recalc
class ItemIn(BaseModel):
    region_id: int | None = Field(None, description="id của món trong kết quả quét; null nếu thêm tay")
    cls: str = Field(description="mã món cuối cùng")
    source: Source = Field(description="model = giữ nguyên, renamed = đã đổi món, manual = thêm tay")
    area_ratio: float | None = Field(None, ge=0, description="lấy từ kết quả quét; dùng để tính lại gram khi đổi món")
    grams: int | None = Field(None, ge=1, le=2000,
                              description="gram người ăn chọn (suất cơm 150/227/300, chỉnh tay 10-1000); "
                                          "bỏ trống = backend tự tính")
    orig_cls: str | None = Field(None, description="mã món mô hình đoán ban đầu (khi renamed)")
    conf: float | None = Field(None, description="độ tin cậy gốc")
    confirmed: bool = Field(True, description="false = món cần xác nhận mà người ăn bỏ qua")


class ItemOut(ItemIn):
    name: str
    grams: int
    kcal: int
    gram_method: str = Field(description="area | rice_mean | rice_depth | class_mean | user")


class RecalcRequest(BaseModel):
    scan_id: str | None = Field(None, description="cần khi có cơm và bật độ sâu, để lấy thể tích cơm")
    items: list[ItemIn]


class RecalcResponse(BaseModel):
    items: list[ItemOut]
    total: Total


# ------------------------------------------------------------------ /api/meals
class MealCreate(BaseModel):
    scan_id: str | None = Field(None, description="null nếu bữa ăn nhập tay hoàn toàn")
    items: list[ItemIn] = Field(min_length=1)
    eaten_at: datetime = Field(description="ISO 8601 có múi giờ, vd. 2026-11-12T12:30:00+07:00")
    share_corrections: bool = Field(False, description="người ăn đồng ý góp ảnh + chỉnh sửa để cải thiện mô hình")


class MealCreated(BaseModel):
    meal_id: int
    total: Total
    correction_saved: bool = Field(description="false nếu không bật góp ảnh hoặc ảnh đã hết hạn")


class MealItemOut(BaseModel):
    cls: str
    name: str
    grams: int
    kcal: int
    source: Source
    orig_cls: str | None
    confirmed: bool


class MealOut(BaseModel):
    meal_id: int
    eaten_at: datetime = Field(description="theo giờ Việt Nam (+07:00)")
    total_g: int
    total_kcal: int
    model_version: str | None
    items: list[MealItemOut]


class DayMeals(BaseModel):
    date: date
    meals: list[MealOut]
    total_kcal: int
    daily_pct: int = Field(description="% so với 2.000 kcal")


# ------------------------------------------------------------------ /api/health
class Health(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_version: str
    gpu: bool
    depth_enabled: bool
    dino_enabled: bool
    engine: str = Field(description="real | mock")
    queue: int = Field(description="số ảnh đang chờ + đang chạy")
    detail: str | None = None
