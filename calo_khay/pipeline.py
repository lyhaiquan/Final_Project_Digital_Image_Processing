"""Calo Khay: ảnh khay cơm -> từng món (mask, độ tin cậy) -> gram -> kcal.

Các bước (giống hệt pipeline dùng để chấm điểm trong báo cáo):
  1. YOLO26s-seg (models/food_seg_v4_final.pt, 45 món): mask từng món.
     Mặc định có thêm bước DINOv2 (RegionClassifier): lấy cả mask conf >= 0.1, phân loại lại từng vùng bằng đặc trưng
     DINOv2 (models/dino_head.npz) kết hợp với điểm của YOLO, khử mask trùng, bỏ mask yếu mà DINO không chắc.
     Tắt bằng use_dino=False -> như cũ: chỉ YOLO, conf >= 0.25.
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
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.normpath(os.path.join(HERE, "..", "models"))
DEPTH_HF = "depth-anything/Depth-Anything-V2-Large-hf"
DINO_HF = "facebook/dinov2-base"
SEG_CONF, PLATE_CONF, IMGSZ = 0.25, 0.1, 640
# bước DINOv2 (tham số chọn trên tập val, xem research/improve/):
SEG_CONF_LOW = 0.1        # lấy cả mask yếu, để DINO quyết định
DINO_W = 0.8              # trọng số log-xác suất DINO khi kết hợp với YOLO
NMS_IOU = 0.7             # hai mask trùng nhau quá mức này -> giữ mask conf cao hơn
DINO_GATE = 0.5           # mask conf < 0.25 chỉ được giữ nếu xác suất kết hợp >= ngưỡng này
FLAG_DINO, FLAG_YOLO = 0.75, 0.5   # dưới ngưỡng -> "cần xác nhận"
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


# ----------------------------------------------------------------------------- phân loại lại bằng DINOv2
def torch_device(device=None) -> str:
    """'cpu' / '0' (kiểu Ultralytics) / None -> tên thiết bị của torch."""
    import torch
    if device is None or device == "":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{device}" if str(device).isdigit() else str(device)


class RegionClassifier:
    """Mỗi vùng món: crop vuông quanh mask (đệm 15%) -> DINOv2-base 224 px -> trung bình các token patch NẰM TRONG mask
    -> hồi quy logistic (models/dino_head.npz, fit trên polygon nhãn tay train+val bằng tools/export_dino_head.py)."""
    S, P, PAD = 224, 14, 0.15
    MEAN, STD = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])

    def __init__(self, models_dir: str = MODELS, device=None, load_head: bool = True):
        import torch
        from transformers import AutoModel
        self.torch, self.dev = torch, torch_device(device)
        self.model = AutoModel.from_pretrained(DINO_HF).to(self.dev).eval()
        self.half = self.dev.startswith("cuda")
        if self.half:
            self.model.half()
        self.G = self.S // self.P
        if load_head:
            h = np.load(os.path.join(models_dir, "dino_head.npz"), allow_pickle=False)
            self.W, self.b, self.names = h["coef"], h["intercept"], [str(x) for x in h["names"]]

    def _crop(self, bgr, mask):
        S, G = self.S, self.G
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = max(x1 - x0, y1 - y0) * (1 + 2 * self.PAD)
        a, b = int(round((x0 + x1) / 2 - side / 2)), int(round((y0 + y1) / 2 - side / 2)); side = int(round(side))
        M = np.float32([[1, 0, -a], [0, 1, -b]])          # phần ngoài ảnh tô xám
        c = cv2.warpAffine(bgr, M, (side, side), borderValue=(124, 116, 104))
        mk = cv2.warpAffine(mask.astype(np.uint8), M, (side, side))
        c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA if side > S else cv2.INTER_CUBIC)
        mk = cv2.resize(mk.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA)
        if mk.sum() < 1e-6:
            mk[:] = 1
        return (c[:, :, ::-1].astype(np.float32) / 255 - self.MEAN) / self.STD, mk / mk.sum()

    def features(self, bgr: np.ndarray, masks, bs: int = 16) -> np.ndarray:
        """[N, 768], đã chuẩn hoá L2."""
        torch = self.torch
        masks = [m for m in masks]
        if not masks:
            return np.zeros((0, 768), np.float32)
        xs, ws = zip(*[self._crop(bgr, m) for m in masks])
        x, w = np.stack(xs).transpose(0, 3, 1, 2), np.stack(ws).reshape(len(xs), -1, 1)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), bs):
                xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(self.dev)
                t = self.model(pixel_values=xb.half() if self.half else xb).last_hidden_state.float()
                out.append((t[:, -self.G * self.G:] * torch.from_numpy(w[i:i + bs]).to(self.dev)).sum(1).cpu().numpy())
        f = np.concatenate(out)
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)

    def proba(self, bgr: np.ndarray, masks) -> np.ndarray:
        """[N, số lớp] theo thứ tự self.names."""
        z = self.features(bgr, masks) @ self.W.T + self.b
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)


def relabel(cls: np.ndarray, conf: np.ndarray, P: np.ndarray, masks: np.ndarray):
    """Kết hợp DINO với YOLO rồi lọc. -> (lớp mới [N], giữ [N] bool, xác suất kết hợp [N, K])."""
    n, K = P.shape
    if n == 0:
        return cls, np.zeros(0, bool), P
    py = np.repeat(((1 - conf) / (K - 1))[:, None], K, 1); py[np.arange(n), cls] = conf   # YOLO chỉ cho điểm lớp nó chọn
    lp = DINO_W * np.log(P + 1e-6) + (1 - DINO_W) * np.log(py + 1e-6)
    q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    new, pm = q.argmax(1), q.max(1)
    keep = (conf >= SEG_CONF) | (pm >= DINO_GATE)
    # khử mask trùng, không phân biệt lớp (IoU tính trên mask thu nhỏ cho nhanh)
    H, W = masks.shape[1:]; s = 256 / max(H, W)
    small = np.stack([cv2.resize(m.astype(np.uint8), (max(1, round(W * s)), max(1, round(H * s))), interpolation=cv2.INTER_NEAREST)
                      for m in masks]).reshape(n, -1).astype(np.float32)
    area = small.sum(1); inter = small @ small.T
    iou = inter / np.maximum(area[:, None] + area[None] - inter, 1)
    order = np.argsort(-conf, kind="stable")
    for a, i in enumerate(order):
        if keep[i]:
            for j in order[a + 1:]:
                if iou[i, j] >= NMS_IOU:
                    keep[j] = False
    return new, keep, q


# ----------------------------------------------------------------------------- gram
def _loglog(m: dict, cls: str, x: float) -> float:
    a = m["intercept"][cls] if cls in m["intercept"] else m["intercept"]["_other"]
    return float(np.exp(a + m["slope"] * np.log(max(x, 1e-6))) * m["smearing"])


def estimate_grams(gm: dict, cls: str, A: float | None, V: float | None = None) -> tuple[float, str]:
    """Công thức gram (QT-03, QT-04) tách khỏi CaloKhay để backend tính lại mà không cần nạp model.
    gm: nội dung gram_model.json. Trả về (gram, cách tính)."""
    if cls == "rice":
        if V is not None and V > 0:
            return _loglog(gm["rice_depth"], "rice", V), "rice_depth"
        return gm["rice_mean_g"], "rice_mean"
    if A is None:  # không thấy đĩa -> khẩu phần TB của món
        return gm["class_mean_g"].get(cls, gm["global_mean_g"]), "class_mean"
    return _loglog(gm["area"], cls, A), "area"


class CaloKhay:
    def __init__(self, use_depth: bool = False, device: str | None = None, models_dir: str = MODELS, use_dino: bool = True):
        from ultralytics import YOLO, YOLOE
        self.device = device
        self.use_dino = use_dino
        self.dino = RegionClassifier(models_dir, device) if use_dino else None
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
            self.dev = torch_device(device)
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
        return estimate_grams(self.gm, cls, A, V)

    def analyze(self, image, flag_conf: float | None = None, use_depth: bool | None = None) -> dict:
        """flag_conf: món có độ tin cậy dưới ngưỡng này -> needs_confirm. None = mặc định (0.75 khi có DINO, 0.5 khi không).
        use_depth: bật/tắt độ sâu cho riêng lần này; None = theo lúc khởi tạo. Chỉ bật được khi đã nạp model độ sâu."""
        if flag_conf is None:
            flag_conf = FLAG_DINO if self.use_dino else FLAG_YOLO
        use_depth = self.use_depth if use_depth is None else (use_depth and self.use_depth)
        timing = {}
        t0 = time.perf_counter()
        bgr = cv2.imread(image) if isinstance(image, str) else image
        if bgr is None:
            raise FileNotFoundError(image)
        H, W = bgr.shape[:2]
        warnings = []

        # 1) phân đoạn món
        r = self.seg.predict(bgr, conf=SEG_CONF_LOW if self.use_dino else SEG_CONF, imgsz=IMGSZ, retina_masks=True,
                             verbose=False, device=self.device)[0]
        dets = []
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5
            cls, conf = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy().astype(float)
            ok = masks.reshape(len(masks), -1).any(1)
            masks, cls, conf = masks[ok], cls[ok], conf[ok]
            if self.use_dino and len(masks):
                # xác suất DINO theo thứ tự lớp của YOLO
                Pd = self.dino.proba(bgr, masks); P = np.zeros((len(masks), len(self.seg.names)))
                idx = {n: i for i, n in self.seg.names.items()}
                for j, n in enumerate(self.dino.names):
                    P[:, idx[n]] = Pd[:, j]
                cls, keep, q = relabel(cls, conf, P, masks)
                for mk, c, qi, k in zip(masks, cls, q, keep):
                    if k:
                        alt = [self.seg.names[int(a)] for a in np.argsort(-qi)[1:3] if qi[a] >= 0.1]
                        dets.append(dict(cls=self.seg.names[int(c)], conf=float(qi[c]), mask=mk, polys=mask_polygons(mk), alt=alt))
            else:
                for mk, c, cf in zip(masks, cls, conf):
                    dets.append(dict(cls=self.seg.names[int(c)], conf=float(cf), mask=mk, polys=mask_polygons(mk), alt=[]))
        if not dets:
            warnings.append("no_food")
        timing["segment"] = time.perf_counter() - t0; t0 = time.perf_counter()
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
        timing["plate"] = time.perf_counter() - t0; t0 = time.perf_counter()

        # gộp theo món (mỗi món một dòng)
        by_cls: dict[str, dict] = {}
        for d in dets:
            e = by_cls.setdefault(d["cls"], dict(mask=np.zeros((H, W), bool), conf=0.0, polygons=[], alt=[]))
            e["mask"] |= d["mask"]; e["polygons"] += d["polys"]
            if d["conf"] > e["conf"]:
                e["conf"], e["alt"] = d["conf"], d["alt"]

        # 3) độ sâu cho cơm
        V = None
        if use_depth and "rice" in by_cls and exposed is not None:
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
        timing["depth"] = time.perf_counter() - t0; t0 = time.perf_counter()

        # 4) gram, kcal
        items = []
        for i, (cls, e) in enumerate(sorted(by_cls.items(), key=lambda kv: -kv[1]["mask"].sum())):
            area_px = int(e["mask"].sum())
            A = area_px / plate_px if plate_px else None
            g, how = self.grams(cls, A, V if cls == "rice" else None)
            g = round(g)  # kcal tính từ gram đã làm tròn -> khớp với số hiển thị và với /api/recalc
            items.append(dict(
                id=i + 1, cls=cls, name=self.name_vi.get(cls, cls), conf=round(e["conf"], 3),
                needs_confirm=e["conf"] < flag_conf,
                # gợi ý: lớp DINO xếp thứ 2-3 (nếu có), rồi tới các món hay bị nhầm theo bảng nhầm lẫn
                suggestions=list(dict.fromkeys(e["alt"] + self.gm.get("confusions", {}).get(cls, [])))[:3],
                area_px=area_px, area_ratio=None if A is None else round(A, 4),
                grams=round(g), gram_method=how, kcal=round(g * self.kcal100.get(cls, 0) / 100),
                polygons=e["polygons"]))
        tg, tk = sum(i["grams"] for i in items), sum(i["kcal"] for i in items)
        timing["grams"] = time.perf_counter() - t0
        return dict(
            timing_ms={k: round(v * 1000) for k, v in timing.items()},
            image=dict(w=W, h=H), plate_found=plate_px is not None,
            plate_px=None if plate_px is None else round(plate_px), plate_ellipse=ellipse is not None,
            rice_volume=V, items=items,
            total=dict(grams=tg, kcal=tk, error_pct=ERROR_PCT, daily_pct=round(100 * tk / DAILY_KCAL)),
            warnings=warnings)
