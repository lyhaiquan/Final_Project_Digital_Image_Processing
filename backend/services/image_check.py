"""Kiểm tra ảnh đầu vào (QT-07, FR-B01, FR-B02): đọc theo nội dung file, xoay EXIF, đo độ nét."""
from __future__ import annotations

import io
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..errors import ApiError

ALLOWED = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = 60_000_000  # chặn "bom giải nén" (ảnh nhỏ nhưng khai báo kích thước khổng lồ)


def decode(data: bytes, min_side: int) -> np.ndarray:
    """bytes -> ảnh BGR đã xoay theo EXIF. Lỗi -> ApiError 400."""
    if not data:
        raise ApiError(400, "invalid_image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data))
            fmt = im.format
            if fmt not in ALLOWED:
                raise ApiError(400, "invalid_image")
            im.load()
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
    except ApiError:
        raise
    except Exception:  # file hỏng, PDF đổi đuôi, ảnh quá nhiều pixel...
        raise ApiError(400, "invalid_image")
    if min(im.size) < min_side:
        raise ApiError(400, "image_too_small")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def sharpness(bgr: np.ndarray) -> float:
    """Phương sai Laplacian trên ảnh xám thu về cạnh dài 1024 px (để không phụ thuộc độ phân giải)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 1024 / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
