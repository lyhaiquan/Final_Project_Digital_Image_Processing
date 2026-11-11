"""Thử nghiệm 1: giữ mask của YOLO, phân loại lại món bằng đặc trưng mô hình nền.
CHỌN trên val (model chỉ-train, tham khảo = polygon train). Test chỉ chạy khi thêm cờ --test.

    python eval_cls.py dinov2b [--test]
"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from common import *

key = sys.argv[1] if len(sys.argv) > 1 else "dinov2b"
L = lambda n: np.load(f"{CACHE}/emb_{key}/{n}.npz", allow_pickle=True)


def feat(z, kind):
    c, m = normalize(z["cls"]), normalize(z["mp"])
    return c if kind == "cls" else m if kind == "mp" else np.hstack([c, m])


def fit(X, y, C):
    return LogisticRegression(C=C, max_iter=3000).fit(X, y)


def proba(clf, X):
    p = np.zeros((len(X), K)); p[:, clf.classes_] = clf.predict_proba(X); return p


def knn(Xr, yr, Xq, k=5):
    s = Xq @ Xr.T; idx = np.argsort(-s, 1)[:, :k]
    p = np.zeros((len(Xq), K))
    for i in range(len(Xq)):
        for j in idx[i]: p[i, yr[j]] += s[i, j]
    return p / p.sum(1, keepdims=True)


def relabel(tag, split, P, w=1.0):
    """w=1: thay hẳn nhãn YOLO; w<1: cộng log-xác suất với YOLO (YOLO: conf cho lớp nó chọn, phần còn lại chia đều)"""
    ims = L(f"pred_{tag}")["im"]; cd = covdata(tag, split)

    def cls_of(im):
        _, _, c, s = cd[im]; p = P[ims == im]
        if len(c) == 0: return c
        if w < 1:
            py = np.repeat(((1 - s) / (K - 1))[:, None], K, 1); py[np.arange(len(c)), c] = s
            p = w * np.log(p + 1e-6) + (1 - w) * np.log(py + 1e-6)
        return p.argmax(1)
    return cls_of


def run(split, tags, ref_splits, cfg):
    kind, clfname, par = cfg
    R = [L(f"gt_{s}") for s in ref_splits]
    Xr = np.vstack([feat(r, kind) for r in R]); yr = np.concatenate([r["y"] for r in R])
    clf = fit(Xr, yr, par) if clfname == "lr" else None
    pr = (lambda X: proba(clf, X)) if clf is not None else (lambda X: knn(Xr, yr, X, par))
    g = L(f"gt_{split}"); acc_gt = (pr(feat(g, kind)).argmax(1) == g["y"]).mean() * 100
    out = {}
    for tag in tags:
        P = pr(feat(L(f"pred_{tag}"), kind))
        out[tag] = P
    return acc_gt, out


def table(split, tags, ref_splits, cfgs, fuse_w=(1.0,)):
    rows = []
    base = [score(split, t) for t in tags]
    rows.append(dict(cach="YOLO (hiện tại)", acc_maskGT=np.nan, **{k: np.mean([b[k] for b in base]) for k in ("dung", "nham", "sot", "acc", "P", "R", "F1", "fp")}))
    for cfg in cfgs:
        a, Ps = run(split, tags, ref_splits, cfg)
        for w in fuse_w:
            sc = [score(split, t, cls_of=relabel(t, split, Ps[t], w)) for t in tags]
            rows.append(dict(cach=f"{cfg[0]}+{cfg[1]}{cfg[2]} w={w}", acc_maskGT=a, **{k: np.mean([b[k] for b in sc]) for k in ("dung", "nham", "sot", "acc", "P", "R", "F1", "fp")}))
    return pd.DataFrame(rows).round(1)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    a = [x for x in sys.argv[2:] if x != "--test"]
    if a:   # cấu hình cụ thể: <kind> <lr|knn> <tham số> [w ...]
        cfgs = [(a[0], a[1], float(a[2]) if a[1] == "lr" else int(a[2]))]; ws = tuple(float(x) for x in a[3:]) or (1.0,)
    else:
        cfgs = [(k, "lr", C) for k in ("cls", "mp", "both") for C in (1, 10, 100)] + [(k, "knn", n) for k in ("cls", "mp", "both") for n in (1, 5)]; ws = (1.0,)
    if "--test" not in sys.argv:
        print(f"== VAL, {key}: 3 model chỉ-train, tham khảo = polygon train (trung bình 3 seed)")
        print(table("val", ("tr0", "tr1", "tr2"), ("train",), cfgs, ws).to_string(index=False))
    else:
        print(f"== TEST, {key}: 3 model full, tham khảo = polygon train+val (trung bình 3 seed)")
        print(table("test", ("full0", "full1", "full2"), ("train", "val"), cfgs, ws).to_string(index=False))
