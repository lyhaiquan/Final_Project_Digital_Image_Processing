"""Hướng 6: cờ "cần xác nhận". Đơn vị = món hiển thị (ảnh, lớp). Lỗi = món không có trong nhãn của ảnh (món "ma").
Chọn loại điểm + ngưỡng trên VAL (model chỉ-train), chấm trên TEST (model full)."""
import sys, numpy as np, pandas as pd
from common import *
import eval_cls as E, postproc as PP
E.key = "dinov2b224"; CFG = dict(w=0.8, t_low=0.1, nms=0.7, gate=0.5)


def items(split, tags, ref, new):
    _, Ps = E.run(split, tags, ref, ("mp", "lr", 100.0)); rows = []
    for t in tags:
        cd = covdata(t, split); pi = pairiou(t, split); ims = E.L(f"pred_{t}")["im"]
        for im, (g, cv, c, s) in cd.items():
            P = Ps[t][ims == im]
            if new: cls, keep, pm = PP.decide(c, s, P, pi[im][0], **CFG)
            else: cls, keep, pm = c, s >= 0.25, s
            pd_ = P[np.arange(len(c)), cls] if len(c) else np.zeros(0)
            for ci in set(cls[keep].tolist()):
                k = keep & (cls == ci)
                rows.append(dict(tag=t, im=im, cls=ci, err=ci not in set(g.tolist()), yolo=s[k].max(), fused=pm[k].max(), dino=pd_[k].max(),
                                 both=min(s[k].max(), pd_[k].max()), prod=(s[k] * pd_[k]).max()))
    return pd.DataFrame(rows)


def auc(score, err):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1, n0 = err.sum(), (~err).sum()
    return 1 - (r[err].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)       # điểm thấp -> lỗi


def rep(d, col, thr):
    f = d[col] < thr
    return dict(co=round(100 * f.mean(), 1), bat_loi=round(100 * (f & d.err).sum() / max(d.err.sum(), 1), 1), co_dung=round(100 * d.err[f].mean(), 1), n=len(d), loi=int(d.err.sum()))


va_old, va_new = items("val", ("tr0", "tr1", "tr2"), ("train",), False), items("val", ("tr0", "tr1", "tr2"), ("train",), True)
te_old, te_new = items("test", ("full0", "full1", "full2"), ("train", "val"), False), items("test", ("full0", "full1", "full2"), ("train", "val"), True)
print("AUC phân biệt món sai (val):  cũ/yolo", round(auc(va_old.yolo.values, va_old.err.values), 3),
      {c: round(auc(va_new[c].values, va_new.err.values), 3) for c in ("yolo", "fused", "dino", "both", "prod")})
best = max(("yolo", "fused", "dino", "both", "prod"), key=lambda c: auc(va_new[c].values, va_new.err.values))
rows = [dict(tap="val", cach="cũ: conf YOLO < 0.5", **rep(va_old, "yolo", 0.5)), dict(tap="test", cach="cũ: conf YOLO < 0.5", **rep(te_old, "yolo", 0.5))]
for q in (0.2, 0.25, 0.3, 0.4):
    thr = float(np.quantile(va_new[best], q))
    rows.append(dict(tap="val", cach=f"mới: {best} < {thr:.2f}", **rep(va_new, best, thr))); rows.append(dict(tap="test", cach=f"mới: {best} < {thr:.2f}", **rep(te_new, best, thr)))
print("AUC test: cũ", round(auc(te_old.yolo.values, te_old.err.values), 3), "| mới", round(auc(te_new[best].values, te_new.err.values), 3))
pd.set_option("display.width", 200); print(pd.DataFrame(rows).sort_values("tap", ascending=False, kind="stable").to_string(index=False))
