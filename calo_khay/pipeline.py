"""Calo Khay: ảnh khay cơm -> từng món (mask, độ tin cậy) -> gram -> kcal.

Các bước (giống hệt pipeline dùng để chấm điểm trong báo cáo):
  1. YOLO26s-seg (models/food_seg_v4_final.pt, 45 món): mask từng món, conf >= 0.25.
  2. YOLOE (models/yoloe_plate.pt, đã gắn sẵn chữ "plate"): tìm đĩa chứa đồ ăn -> diện tích đĩa (pixel),
     fit elip nếu đĩa bị khung ảnh cắt. Diện tích đĩa là "thước đo" để khử khoảng cách chụp.
  3. Món khác cơm:  gram = exp(a_món + b·log(diện tích món / diện tích đĩa)) · smearing   (models/gram_model.json)
  4. Cơm: mặc định = khẩu phần trung bình (~228 g). Nếu bật depth: Depth-Anything-V2 Large -> độ cao so với mặt đĩa
     -> "thể tích" tương đối V -> gram = exp(a + b·log V) · smearing.
  5. kcal = gram × kcal/100 g (models/calories.csv).
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.normpath(os.path.join(HERE, "..", "models"))
DEPTH_HF = "depth-anything/Depth-Anything-V2-Large-hf"
SEG_CONF, PLATE_CONF, IMGSZ = 0.25, 0.1, 640
ERROR_PCT = 30            # sai số hiển thị cho mỗi món (QT-06)
DAILY_KCAL = 2000         # nhu cầu tham chiếu một ngày (QT-11)


# ----------------------------------------------------------------------------- đĩa
def plate_area(hull_mask: np.ndarray, margin: int = 4):
    """Diện tích đĩa (pixel). Nếu > 2% viền nằm sát mép ảnh (đĩa bị cắt) thì fit elip lên phần viền còn lại;
    chỉ nhận elip nếu nó lớn hơn bao lồi và nhỏ hơn 2 lần bao lồi."""
    H, W = hull_mask.shape
    m = (hull_mask > 0).astype(np.uint8)
    c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    hm = np.zeros((H, W), np.uint8); cv2.fillPoly(hm, [cv2.convexHull(c)], 1)
    hull_a = float(hm.sum())
    dense = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    keep = (dense[:, 0] > margin) & (dense[:, 0] < W - 1 - margin) & (dense[:, 1] > margin) & (dense[:, 1] < H - 1 - margin)
    if 1 - keep.mean() > 0.02 and keep.sum() >= 50:
        el = cv2.fitEllipse(dense[keep].astype(np.float32))
        ea = np.pi * el[1][0] * el[1][1] / 4
        if hull_a <= ea <= 2 * hull_a:
            return float(ea), el
    return hull_a, None


def mask_polygons(mask: np.ndarray, min_frac: float = 0.05) -> list:
    """Viền ngoài của mask -> danh sách đa giác [[x, y], ...] (mỗi mảnh rời một đa giác, bỏ mảnh vụn)."""
    cs = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not cs:
        return []
    big = max(cv2.contourArea(c) for c in cs)
    out = []
    for c in cs:
        if cv2.contourArea(c) >= min_frac * big and len(c) >= 3:
            c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
            if len(c) >= 3:
                out.append(c.reshape(-1, 2).tolist())
    return out


def convex_fill(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, np.uint8)
    cs = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if cs:
        cv2.fillPoly(out, [cv2.convexHull(np.vstack(cs))], 1)
    return out > 0


# ----------------------------------------------------------------------------- độ sâu
def fit_plane(d: np.ndarray, ring: np.ndarray, iters: int = 4):
    """Mặt phẳng bền vững (lọc MAD lặp) trên các điểm mặt đĩa lộ ra."""
    ys, xs = np.nonzero(ring)
    if len(xs) < 200:
        return None
    if len(xs) > 40000:
        sel = np.random.default_rng(0).choice(len(xs), 40000, replace=False); ys, xs = ys[sel], xs[sel]
    z = d[ys, xs].astype(np.float64); A = np.c_[xs, ys, np.ones_like(xs)].astype(np.float64)
    keep = np.ones(len(z), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef; mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        keep = np.abs(r) < 3 * 1.4826 * mad
    H, W = d.shape; yy, xx = np.mgrid[0:H, 0:W]
    return coef[0] * xx + coef[1] * yy + coef[2]


# ----------------------------------------------------------------------------- gram
def _loglog(m: dict, cls: str, x: float) -> float:
    a = m["intercept"][cls] if cls in m["intercept"] else m["intercept"]["_other"]
    return float(np.exp(a + m["slope"] * np.log(max(x, 1e-6))) * m["smearing"])


class CaloKhay:
    def __init__(self, use_depth: bool = False, device: str | None = None, models_dir: str = MODELS):
        from ultralytics import YOLO, YOLOE
        self.device = device
        self.seg = YOLO(os.path.join(models_dir, "food_seg_v4_final.pt"))
        self.plate = YOLOE(os.path.join(models_dir, "yoloe_plate.pt"))
        self.gm = json.load(open(os.path.join(models_dir, "gram_model.json"), encoding="utf-8"))
        cal = pd.read_csv(os.path.join(models_dir, "calories.csv"))
        self.kcal100 = dict(zip(cal["class"], cal.kcal_per_100g.fillna(0.0)))
        self.name_vi = dict(zip(cal["class"], cal.name_vi))
        self.use_depth = use_depth
        if use_depth:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.torch = torch
            self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.dproc = AutoImageProcessor.from_pretrained(DEPTH_HF)
            self.dmodel = AutoModelForDepthEstimation.from_pretrained(DEPTH_HF).to(self.dev).eval()

    # --------------------------------------------------------------------- các bước
    def _depth(self, bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        H, W = bgr.shape[:2]
        with torch.no_grad():
            inp = self.dproc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors="pt").to(self.dev)
            pd_ = self.dmodel(**inp).predicted_depth
            d = torch.nn.functional.interpolate(pd_.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False)
        return d[0, 0].float().cpu().numpy()

    def grams(self, cls: str, A: float | None, V: float | None = None) -> tuple[float, str]:
        """Trả về (gram, cách tính)."""
        gm = self.gm
        if cls == "rice":
            if V is not None and V > 0:
                return _loglog(gm["rice_depth"], "rice", V), "rice_depth"
            return gm["rice_mean_g"], "rice_mean"
        if A is None:  # không thấy đĩa -> khẩu phần TB của món
            return gm["class_mean_g"].get(cls, gm["global_mean_g"]), "class_mean"
        return _loglog(gm["area"], cls, A), "area"

    def analyze(self, image, flag_conf: float = 0.5) -> dict:
        bgr = cv2.imread(image) if isinstance(image, str) else image
        if bgr is None:
            raise FileNotFoundError(image)
        H, W = bgr.shape[:2]
        warnings = []

        # 1) phân đoạn món
        r = self.seg.predict(bgr, conf=SEG_CONF, imgsz=IMGSZ, retina_masks=True, verbose=False, device=self.device)[0]
        dets = []
        if r.masks is not None:
            for mk, c, cf in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.conf.cpu().numpy()):
                dets.append(dict(cls=self.seg.names[c], conf=float(cf), mask=mk, polys=mask_polygons(mk)))
        if not dets:
            warnings.append("no_food")
        food = np.zeros((H, W), bool)
        for d in dets:
            food |= d["mask"]

        # 2) đĩa
        q = self.plate.predict(bgr, conf=PLATE_CONF, retina_masks=True, verbose=False, device=self.device)[0]
        plates = list(q.masks.data.cpu().numpy() > 0.5) if q.masks is not None else []
        hulls = [convex_fill(p) for p in plates]
        ov = [(h & food).sum() / max(food.sum(), 1) for h in hulls]
        plate_px, ellipse, exposed = None, None, None
        if plates and max(ov) > 0.3:
            k = int(np.argmax(ov))
            plate_px, ellipse = plate_area(hulls[k])
            exposed = plates[k]
        else:
            warnings.append("no_plate")

        # gộp theo món (mỗi món một dòng)
        by_cls: dict[str, dict] = {}
        for d in dets:
            e = by_cls.setdefault(d["cls"], dict(mask=np.zeros((H, W), bool), conf=0.0, polygons=[]))
            e["mask"] |= d["mask"]; e["conf"] = max(e["conf"], d["conf"]); e["polygons"] += d["polys"]

        # 3) độ sâu cho cơm
        V = None
        if self.use_depth and "rice" in by_cls and exposed is not None:
            d = self._depth(bgr)
            ring = cv2.erode(exposed.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            plane = fit_plane(d, ring)
            if plane is None:
                warnings.append("depth_plane_failed")
            else:
                hull = convex_fill(exposed)
                spread = float(np.percentile(d[hull], 95) - np.percentile(d[hull], 5)) + 1e-6
                h = (d - plane) / spread
                V = float(np.clip(h[by_cls["rice"]["mask"]], 0, None).sum()) / plate_px

        # 4) gram, kcal
        items = []
        for i, (cls, e) in enumerate(sorted(by_cls.items(), key=lambda kv: -kv[1]["mask"].sum())):
            area_px = int(e["mask"].sum())
            A = area_px / plate_px if plate_px else None
            g, how = self.grams(cls, A, V if cls == "rice" else None)
            items.append(dict(
                id=i + 1, cls=cls, name=self.name_vi.get(cls, cls), conf=round(e["conf"], 3),
                needs_confirm=e["conf"] < flag_conf,
                suggestions=self.gm.get("confusions", {}).get(cls, []),
                area_px=area_px, area_ratio=None if A is None else round(A, 4),
                grams=round(g), gram_method=how, kcal=round(g * self.kcal100.get(cls, 0) / 100),
                polygons=e["polygons"]))
        tg, tk = sum(i["grams"] for i in items), sum(i["kcal"] for i in items)
        return dict(
            image=dict(w=W, h=H), plate_found=plate_px is not None,
            plate_px=None if plate_px is None else round(plate_px), plate_ellipse=ellipse is not None,
            rice_volume=V, items=items,
            total=dict(grams=tg, kcal=tk, error_pct=ERROR_PCT, daily_pct=round(100 * tk / DAILY_KCAL)),
            warnings=warnings)
