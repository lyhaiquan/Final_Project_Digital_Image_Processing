"""Dùng chung cho các thử nghiệm cải thiện (không train lại YOLO).

Quy ước: mọi mask/polygon được đưa về "độ phân giải làm việc" (cạnh dài <= 1280).
Model chỉ-train (tr0..2) chấm trên val để CHỌN cách; model full (full0..2) chấm trên test MỘT lần.
"""
import os, sys, collections
import numpy as np, pandas as pd, cv2

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
D = "D:/food_dataset_v4"
ROOT = "D:/food-labeling"
CACHE = f"{ROOT}/improve/cache"
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
K = len(NAMES)
MODELS = {
    "tr0": f"{ROOT}/kaggle/out_v4b/final_s0_best.pt",
    "tr1": f"{ROOT}/kaggle/out_v4b/final_s1_best.pt",
    "tr2": f"{ROOT}/kaggle/out_v4b/final_s2_best.pt",
    "full0": f"{ROOT}/kaggle/out_v4full_s02/full_s0_last.pt",
    "full1": f"{ROOT}/kaggle/out_v4full/full_s1_last.pt",      # = models/food_seg_v4_final.pt
    "full2": f"{ROOT}/kaggle/out_v4full_s02/full_s2_last.pt",
}
MAXSIDE = 1280
MAN = pd.read_csv(f"{D}/manifest.csv")


def images(split):
    return sorted(MAN[MAN.split == split].image)


def read(split, im):
    """ảnh BGR ở độ phân giải làm việc"""
    a = cv2.imread(f"{D}/images/{split}/{im}")
    s = MAXSIDE / max(a.shape[:2])
    if s < 1: a = cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return a


def gt(split, im, H, W):
    """[(lớp:int, mask bool HxW)] cho từng polygon nhãn tay"""
    out = []
    for l in open(f"{D}/labels/{split}/{im[:-4]}.txt"):
        t = l.split()
        if not t: continue
        xy = (np.array(t[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [xy], 1)
        out.append((int(t[0]), m > 0))
    return out


def save_pred(tag, split, im, aug, masks, cls, conf):
    os.makedirs(f"{CACHE}/pred/{tag}_{aug}", exist_ok=True)
    np.savez_compressed(f"{CACHE}/pred/{tag}_{aug}/{im[:-4]}.npz", bits=np.packbits(masks.reshape(len(masks), -1), axis=1),
                        shape=np.array(masks.shape), cls=cls, conf=conf)


def load_pred(tag, split, im, aug="n640"):
    z = np.load(f"{CACHE}/pred/{tag}_{aug}/{im[:-4]}.npz")
    n, H, W = z["shape"]
    m = np.unpackbits(z["bits"], axis=1)[:, :H * W].reshape(n, H, W).astype(bool) if n else np.zeros((0, H, W), bool)
    return m, z["cls"].astype(int), z["conf"].astype(float)


_COV = {}


def covdata(tag, split, aug="n640"):
    """{ảnh: (lớp nhãn tay [G], độ phủ [G,N] = |mask_n ∩ polygon_g| / |polygon_g|, lớp dự đoán [N], conf [N])}; có cache"""
    import pickle
    k = (tag, split, aug)
    if k in _COV: return _COV[k]
    f = f"{CACHE}/cov_{tag}_{aug}.pkl"
    if os.path.exists(f):
        _COV[k] = pickle.load(open(f, "rb")); return _COV[k]
    d = {}
    for im in images(split):
        m, c, s = load_pred(tag, split, im, aug)
        G = gt(split, im, *m.shape[1:])
        cv = np.array([[(mi & gm).sum() / max(gm.sum(), 1) for mi in m] for _, gm in G]).reshape(len(G), len(m))
        d[im] = (np.array([g for g, _ in G]), cv, c, s)
    pickle.dump(d, open(f, "wb")); _COV[k] = d
    return d


def score(split, tag, aug="n640", cls_of=None, conf=0.25, keep_of=None):
    """polygon: đúng / nhầm / bỏ sót (như confusion.py: lớp dự đoán phủ nhiều nhất, >=30% diện tích polygon)
       cặp (ảnh, món): precision / recall / F1 (danh sách món người dùng thấy); fp = số cặp (ảnh, món) "ma".
       cls_of(im) -> mảng lớp mới cho N mask (phân loại lại); keep_of(im) -> mặt nạ giữ/bỏ thay cho ngưỡng conf."""
    ok = wrong = miss = tp = fp = fn = 0
    for im, (g, cv, c, s) in covdata(tag, split, aug).items():
        if cls_of is not None: c = np.asarray(cls_of(im))
        k = keep_of(im) if keep_of is not None else s >= conf
        c, cv = c[k], cv[:, k]
        for gi, row in zip(g, cv):
            best, f = -1, 0.0
            for ci in set(c.tolist()):
                v = row[c == ci].max()
                if v > f: best, f = ci, v
            pred = best if f >= 0.3 else -1
            ok += pred == gi; miss += pred == -1; wrong += pred not in (gi, -1)
        gs, ps = set(g.tolist()), set(c.tolist())
        tp += len(gs & ps); fp += len(ps - gs); fn += len(gs - ps)
    n = ok + wrong + miss
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=int(n), dung=int(ok), nham=int(wrong), sot=int(miss), acc=round(100 * ok / n, 1),
                P=round(100 * P, 1), R=round(100 * R, 1), F1=round(200 * P * R / max(P + R, 1e-9), 1), fp=fp)


_IOU = {}


def pairiou(tag, split, aug="n640"):
    """{ảnh: (IoU [N,N], diện tích [N])} giữa các mask dự đoán của cùng ảnh; có cache"""
    import pickle
    k = (tag, split, aug)
    if k in _IOU: return _IOU[k]
    f = f"{CACHE}/iou_{tag}_{aug}.pkl"
    if os.path.exists(f):
        _IOU[k] = pickle.load(open(f, "rb")); return _IOU[k]
    d = {}
    for im in images(split):
        m, _, _ = load_pred(tag, split, im, aug)
        x = m.reshape(len(m), -1).astype(np.float32); a = x.sum(1)
        inter = x @ x.T
        d[im] = (inter / np.maximum(a[:, None] + a[None] - inter, 1), a)
    pickle.dump(d, open(f, "wb")); _IOU[k] = d
    return d
