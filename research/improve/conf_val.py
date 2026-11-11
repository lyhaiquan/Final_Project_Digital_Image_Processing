"""Cặp món hay nhầm trên VAL (pipeline mới: DINOv2 + hạ ngưỡng, 3 model chỉ-train) + kcal/100g của từng món."""
import collections, numpy as np, pandas as pd
from common import *
import eval_cls as E, postproc as PP
E.key = "dinov2b224"; CFG = dict(w=0.8, t_low=0.1, nms=0.7, gate=0.5)
cal = pd.read_csv("D:/CVision/Final_Project_Digital_Image_Processing/models/calories.csv"); KC = dict(zip(cal["class"], cal.kcal_per_100g)); VI = dict(zip(cal["class"], cal.name_vi))
tags = ("tr0", "tr1", "tr2")
_, Ps = E.run("val", tags, ("train",), ("mp", "lr", 100.0))
cnt = collections.Counter(); tot = collections.Counter()
for t in tags:
    cd = covdata(t, "val"); pi = pairiou(t, "val"); ims = E.L(f"pred_{t}")["im"]
    for im, (g, cv, c, s) in cd.items():
        cls, keep, _ = PP.decide(c, s, Ps[t][ims == im], pi[im][0], **CFG)
        cls, cv2_ = cls[keep], cv[:, keep]
        for gi, row in zip(g, cv2_):
            tot[gi] += 1
            if len(cls):
                best = max(set(cls.tolist()), key=lambda ci: row[cls == ci].max()); f = row[cls == best].max()
                if f >= 0.3 and best != gi: cnt[tuple(sorted((NAMES[gi], NAMES[best])))] += 1
rows = [dict(a=a, b=b, n=n / 3, kcal_a=KC[a], kcal_b=KC[b], vi=f"{VI[a]} / {VI[b]}") for (a, b), n in cnt.most_common(25)]
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(1).to_string(index=False))
print("tổng nhầm/seed:", sum(cnt.values()) / 3)
