"""Hậu xử lý: lấy mọi mask conf >= t_low, phân loại lại (DINO kết hợp YOLO), khử mask trùng không phân biệt lớp,
giữ mask nếu conf YOLO >= 0.25 hoặc DINO đủ chắc. Dò tham số trên VAL (3 model chỉ-train).
    python postproc.py [backbone]
"""
import sys, itertools, numpy as np, pandas as pd
from common import *
import eval_cls as E

E.key = sys.argv[1] if len(sys.argv) > 1 else "dinov2b"


def decide(c, s, P, iou, w=0.8, t_low=0.1, nms=0.5, gate=0.0, t_hi=0.25):
    """-> (lớp mới [N], giữ [N] bool, điểm kết hợp [N])"""
    n = len(c)
    if n == 0: return c, np.zeros(0, bool), np.zeros(0)
    py = np.repeat(((1 - s) / (K - 1))[:, None], K, 1); py[np.arange(n), c] = s
    lp = w * np.log(P + 1e-6) + (1 - w) * np.log(py + 1e-6)
    q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    cls, pm = q.argmax(1), q.max(1)
    keep = (s >= t_low) & ((s >= t_hi) | (pm >= gate))
    if nms:
        for i in np.argsort(-s):           # mask conf cao hơn thắng; mask trùng (IoU >= nms) bị loại
            if not keep[i]: continue
            dup = (iou[i] >= nms) & (s < s[i]) | ((iou[i] >= nms) & (s == s[i]) & (np.arange(n) > i))
            keep[dup] = False
    return cls, keep, pm


def evaluate(split, tags, Ps, **kw):
    out = []
    for t in tags:
        ims = E.L(f"pred_{t}")["im"]; cd = covdata(t, split); pi = pairiou(t, split)
        dec = {im: decide(cd[im][2], cd[im][3], Ps[t][ims == im], pi[im][0], **kw) for im in cd}
        out.append(score(split, t, cls_of=lambda im: dec[im][0], keep_of=lambda im: dec[im][1]))
    return {k: round(float(np.mean([o[k] for o in out])), 1) for k in ("dung", "nham", "sot", "acc", "P", "R", "F1", "fp")}, out


if __name__ == "__main__":
    tags = ("tr0", "tr1", "tr2")
    _, Ps = E.run("val", tags, ("train",), ("mp", "lr", 100.0))
    base = [score("val", t) for t in tags]
    rows = [dict(w="YOLO", t_low=0.25, nms="", gate="", **{k: round(float(np.mean([b[k] for b in base])), 1) for k in ("dung", "nham", "sot", "acc", "P", "R", "F1", "fp")})]
    for w, t_low, nms, gate in itertools.product((0.8, 0.6), (0.25, 0.15, 0.1, 0.05), (None, 0.7, 0.5), (0.0, 0.5, 0.8)):
        if t_low == 0.25 and gate: continue
        rows.append(dict(w=w, t_low=t_low, nms=nms, gate=gate, **evaluate("val", tags, Ps, w=w, t_low=t_low, nms=nms, gate=gate)[0]))
    d = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(d.to_string(index=False))
