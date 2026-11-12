"""Bảng MySQL (mục 2.7 SRS). Chạy được cả trên SQLite để test."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    """Giờ UTC dạng naive (MySQL DATETIME không lưu múi giờ)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Scan(Base):
    """Kết quả một lần quét, giữ tạm 24 giờ để /recalc và /meals dùng lại."""
    __tablename__ = "scans"
    scan_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    device_id: Mapped[str | None] = mapped_column(String(64), index=True)
    result_json: Mapped[dict] = mapped_column(JSON)
    image_path: Mapped[str | None] = mapped_column(String(255))
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class Meal(Base):
    __tablename__ = "meals"
    meal_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(String(64), index=True)
    scan_id: Mapped[str | None] = mapped_column(String(36))
    eaten_at: Mapped[datetime] = mapped_column(DateTime, index=True)  # UTC
    total_g: Mapped[int] = mapped_column(Integer)
    total_kcal: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    items: Mapped[list["MealItem"]] = relationship(back_populates="meal", cascade="all, delete-orphan",
                                                   order_by="MealItem.id")
    correction: Mapped["Correction | None"] = relationship(back_populates="meal", cascade="all, delete-orphan")


class MealItem(Base):
    __tablename__ = "meal_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meal_id: Mapped[int] = mapped_column(ForeignKey("meals.meal_id", ondelete="CASCADE"), index=True)
    region_id: Mapped[int | None] = mapped_column(Integer)  # id của món trong kết quả quét; None = thêm tay
    cls: Mapped[str] = mapped_column(String(64))
    grams: Mapped[int] = mapped_column(Integer)
    kcal: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16))  # model | renamed | manual
    orig_cls: Mapped[str | None] = mapped_column(String(64))
    conf: Mapped[float | None] = mapped_column(Float)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=True)
    meal: Mapped[Meal] = relationship(back_populates="items")


class Correction(Base):
    """Chỉ tạo khi người ăn bật 'Góp ảnh để cải thiện mô hình'."""
    __tablename__ = "corrections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meal_id: Mapped[int] = mapped_column(ForeignKey("meals.meal_id", ondelete="CASCADE"), unique=True)
    image_path: Mapped[str] = mapped_column(String(255))
    original_json: Mapped[dict] = mapped_column(JSON)
    corrected_json: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    meal: Mapped[Meal] = relationship(back_populates="correction")


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        kw = dict(connect_args={"check_same_thread": False})
        if ":memory:" in url or url in ("sqlite://", "sqlite:///"):
            kw["poolclass"] = StaticPool
        return create_engine(url, **kw)
    return create_engine(url, pool_pre_ping=True, pool_recycle=3600)


def make_sessionmaker(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)
