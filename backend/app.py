"""Backend Calo Khay.

Chạy (từ thư mục gốc repo):
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
Tài liệu API: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager

# Không để ultralytics tự "pip install" gói thiếu khi server đang chạy (vd. pi-heif): cài sẵn theo requirements.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import errors
from .config import Settings, get_settings
from .db import Base, make_engine, make_sessionmaker
from .deps import Ctx
from .routers import info, meals, scan
from .services import cleanup
from .services.catalog import Catalog
from .services.inference import Inference

DESCRIPTION = """Người ăn chụp ảnh khay cơm căng tin → nhận diện từng món, ước lượng gram và kcal (±30%/món).

- Mọi lỗi trả `{"error": "<mã>", "message": "<câu tiếng Việt>"}` (bảng 2.6 SRS).
- `/api/meals` cần header `X-Device-Id` (UUID frontend tạo, giữ trong trình duyệt).

Giấy phép: Ultralytics AGPL-3.0, Depth-Anything-V2 Large CC-BY-NC-4.0 — chỉ dùng phi thương mại."""


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s.tmp_dir.mkdir(parents=True, exist_ok=True)
        s.corrections_dir.mkdir(parents=True, exist_ok=True)
        engine = make_engine(s.database_url)
        Base.metadata.create_all(engine)
        inference = Inference(s)
        app.state.ctx = Ctx(settings=s, catalog=Catalog(s.models_dir), inference=inference,
                            sessions=make_sessionmaker(engine))
        inference.start()
        task = asyncio.create_task(cleanup.loop(app.state.ctx.sessions, s.tmp_dir, s.scan_ttl_hours, s.cleanup_interval_s))
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        inference.shutdown()
        engine.dispose()

    app = FastAPI(title="Calo Khay API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    errors.install(app)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Content-Type", "X-Device-Id"])
    for r in (scan.router, meals.router, info.router):
        app.include_router(r, prefix="/api")
    if s.frontend_dist and s.frontend_dist.exists():  # NFR-B15: một lệnh uvicorn phục vụ luôn frontend
        app.mount("/", StaticFiles(directory=s.frontend_dist, html=True), name="frontend")
    return app


app = create_app()
