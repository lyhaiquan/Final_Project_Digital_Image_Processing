"""Bọc CaloKhay: nạp một lần + làm nóng (NFR-B02), một luồng GPU xử lý tuần tự qua hàng chờ (NFR-B03).

engine = "mock": không nạp mô hình, trả mock/scan.json (co giãn polygon theo ảnh gửi lên). Dùng để
frontend làm song song và để chạy test trên máy không có model/GPU.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ..config import Settings
from ..errors import ApiError

log = logging.getLogger("calo_khay")


class Inference:
    def __init__(self, settings: Settings):
        self.s = settings
        self.status = "loading"  # loading | ok | error
        self.detail: str | None = None
        self.model = None
        self.mock: dict | None = None
        self.gpu = False
        self.pending = 0  # đang chạy + đang chờ
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calo-gpu")
        seg = "food_seg_v4_final"
        self.version = settings.model_version or (
            "mock" if settings.engine == "mock" else
            seg + ("+dino" if settings.use_dino else "") + ("+depth" if settings.depth_enabled else ""))

    # ------------------------------------------------------------------ nạp model
    def start(self) -> None:
        """Nạp model ở luồng GPU, không chặn server khởi động; /api/health trả loading tới khi xong."""
        self._executor.submit(self._load)

    def _load(self) -> None:
        t = time.perf_counter()
        try:
            if self.s.engine == "mock":
                self.mock = json.loads(self.s.mock_scan.read_text(encoding="utf-8"))
            else:
                import torch
                from calo_khay import CaloKhay
                self.gpu = bool(torch.cuda.is_available())
                self.model = CaloKhay(use_depth=self.s.depth_enabled, device=self.s.device,
                                      models_dir=str(self.s.models_dir), use_dino=self.s.use_dino)
                img = cv2.imread(str(self.s.warmup_image)) if self.s.warmup_image.exists() else None
                if img is None:
                    img = np.full((960, 960, 3), 200, np.uint8)
                self.model.analyze(img)  # làm nóng: lần đầu chậm (~7 s) vì khởi tạo CUDA, tải trọng số
            self.status = "ok"
            log.info("Đã nạp model %s sau %.1f s (gpu=%s)", self.version, time.perf_counter() - t, self.gpu)
        except Exception as e:  # thiếu file model, hết RAM...
            self.status, self.detail = "error", f"{type(e).__name__}: {e}"
            log.exception("Nạp model thất bại")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ suy luận
    async def analyze(self, bgr: np.ndarray, flag_conf: float | None, depth: bool | None) -> dict:
        if self.status != "ok":
            raise ApiError(503, "model_loading", headers={"Retry-After": "5"})
        if self.pending > self.s.max_queue:  # 1 ảnh đang chạy + tối đa max_queue ảnh chờ
            raise ApiError(503, "busy", headers={"Retry-After": "3"})
        self.pending += 1
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, self._run, bgr, flag_conf, depth)
        finally:
            self.pending -= 1

    def _run(self, bgr: np.ndarray, flag_conf: float | None, depth: bool | None) -> dict:
        if flag_conf is None:
            flag_conf = self.s.flag_conf
        want_depth = self.s.depth_enabled if depth is None else depth
        if self.mock is not None:
            res = self._mock_result(bgr, flag_conf)
        else:
            res = self.model.analyze(bgr, flag_conf=flag_conf, use_depth=want_depth)
        if want_depth and not self.s.depth_enabled:
            res["warnings"].append("depth_unavailable")
        return res

    def _mock_result(self, bgr: np.ndarray, flag_conf: float | None) -> dict:
        res = copy.deepcopy(self.mock)
        res.pop("scan_id", None); res.pop("model_version", None)
        H, W = bgr.shape[:2]
        sx, sy = W / res["image"]["w"], H / res["image"]["h"]
        for it in res["items"]:
            it["polygons"] = [[[round(x * sx), round(y * sy)] for x, y in p] for p in it["polygons"]]
            if flag_conf is not None:
                it["needs_confirm"] = it["conf"] < flag_conf
        res["image"] = dict(w=W, h=H)
        res["timing_ms"] = dict(segment=0, plate=0, depth=0, grams=0)
        return res
