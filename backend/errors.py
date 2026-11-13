"""Lỗi nghiệp vụ có mã (bảng 2.6). Mọi lỗi trả về dạng {"error": "<mã>", "message": "<câu tiếng Việt>"}."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("calo_khay")

MESSAGES = {
    "invalid_image": "Ảnh không hợp lệ, hãy chọn JPG hoặc PNG",
    "image_too_small": "Ảnh quá nhỏ, hãy chụp gần hơn",
    "image_too_large": "Ảnh quá lớn (tối đa 10 MB)",
    "scan_expired": "Kết quả đã hết hạn, hãy quét lại",
    "unknown_class": "Mã món không có trong danh mục",
    "busy": "Máy đang bận, thử lại sau vài giây",
    "model_loading": "Máy đang bận, thử lại sau vài giây",
    "missing_device_id": "Thiếu header X-Device-Id",
    "meal_not_found": "Không tìm thấy bữa ăn",
    "benchmark_missing": "Chưa có số liệu benchmark",
    "validation_error": "Dữ liệu gửi lên không hợp lệ",
    "not_found": "Không tìm thấy",
    "internal": "Lỗi máy chủ",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str | None = None, headers: dict | None = None):
        self.status, self.code = status, code
        self.message = message or MESSAGES.get(code, code)
        self.headers = headers


def _body(code: str, message: str, **extra) -> dict:
    return {"error": code, "code": code, "message": message, **extra}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(_: Request, e: ApiError):
        return JSONResponse(_body(e.code, e.message), status_code=e.status, headers=e.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = [{"loc": list(x.get("loc", [])), "msg": x.get("msg")} for x in e.errors()]
        return JSONResponse(_body("validation_error", MESSAGES["validation_error"], details=details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        code = "not_found" if e.status_code == 404 else f"http_{e.status_code}"
        return JSONResponse(_body(code, str(e.detail)), status_code=e.status_code)

    @app.exception_handler(Exception)
    async def _any(request: Request, e: Exception):
        log.exception("Lỗi không mong đợi tại %s", request.url.path)
        return JSONResponse(_body("internal", MESSAGES["internal"]), status_code=500)
