"""GET /api/classes, /api/benchmark, /api/health."""
from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..deps import CtxDep
from ..errors import ApiError
from ..schemas import Error, FoodClass, Health

router = APIRouter(tags=["info"])


@router.get("/classes", response_model=list[FoodClass], summary="Danh mục 45 món")
def classes(ctx: CtxDep):
    return ctx.catalog.classes()


@router.get("/benchmark", responses={404: {"model": Error}},
            summary="Số liệu trang Benchmark (models/benchmark.json, sinh bằng tools/build_benchmark_json.py)")
def benchmark(ctx: CtxDep):
    f = ctx.settings.models_dir / "benchmark.json"
    if not f.exists():
        raise ApiError(404, "benchmark_missing")
    return json.loads(f.read_text(encoding="utf-8"))


@router.get("/health", response_model=Health, responses={503: {"model": Health}},
            summary="Tình trạng server; chỉ trả 200 sau khi nạp và làm nóng model xong")
def health(ctx: CtxDep):
    inf, s = ctx.inference, ctx.settings
    body = Health(status=inf.status, model_version=inf.version, gpu=inf.gpu, depth_enabled=s.depth_enabled,
                  dino_enabled=s.use_dino and s.engine != "mock", engine=s.engine, queue=inf.pending, detail=inf.detail)
    return JSONResponse(body.model_dump(), status_code=200 if inf.status == "ok" else 503)
