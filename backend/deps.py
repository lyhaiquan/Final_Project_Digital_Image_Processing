"""Dependency dùng chung cho các router."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Iterator

from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .db import Scan, utcnow
from .errors import ApiError
from .services.catalog import Catalog
from .services.inference import Inference

DEVICE_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@dataclass
class Ctx:
    settings: Settings
    catalog: Catalog
    inference: Inference
    sessions: sessionmaker


def get_ctx(request: Request) -> Ctx:
    return request.app.state.ctx


def get_db(ctx: Annotated[Ctx, Depends(get_ctx)]) -> Iterator[Session]:
    with ctx.sessions() as db:
        yield db


def device_id(x_device_id: Annotated[str | None, Header(description="mã thiết bị ngẫu nhiên (UUID) frontend tạo và giữ")] = None) -> str:
    if not x_device_id or not DEVICE_RE.match(x_device_id):
        raise ApiError(400, "missing_device_id")
    return x_device_id


def optional_device_id(x_device_id: Annotated[str | None, Header()] = None) -> str | None:
    return x_device_id if x_device_id and DEVICE_RE.match(x_device_id) else None


def live_scan(db: Session, scan_id: str | None, now: datetime | None = None) -> Scan | None:
    """Scan còn hạn, hoặc None nếu không truyền scan_id. Hết hạn / không có -> None (người gọi tự quyết lỗi)."""
    if not scan_id:
        return None
    s = db.get(Scan, scan_id)
    if s is None or s.expires_at < (now or utcnow()):
        return None
    return s


CtxDep = Annotated[Ctx, Depends(get_ctx)]
DbDep = Annotated[Session, Depends(get_db)]
DeviceDep = Annotated[str, Depends(device_id)]
