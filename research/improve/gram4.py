"""Chọn lam theo CV với quy tắc 1 sai số chuẩn (lam lớn nhất có MAE CV <= MAE nhỏ nhất + 1 SE theo buổi chụp)."""
import sys, numpy as np, pandas as pd
from common import *
sys.path.insert(0, ROOT)
import depth_eval_pred as DP, gram2 as G2
f = DP.it.merge(DP.F[["image", "class", "area_px"]], on=["image", "class"], how="left")
f["A"] = f.area_px / f.image.map(DP.pl.plate_px)
f = f[f.area_px.notna() & (f["class"] != "rice") & f.A.notna()]
fs = f[f.split != "test"]; groups = sorted(fs.group.unique())
L = (0.3, 1, 2, 3, 5, 10, 20, 30, 100)
E = np.array([[np.abs(G2.fit_predict(fs[fs.group != g], fs[fs.group == g], l, False) - fs[fs.group == g].grams.values).mean() for g in groups] for l in L])
n = np.array([(fs.group == g).sum() for g in groups])
m = (E * n).sum(1) / n.sum(); k = m.argmin()
se = np.sqrt(((E[k] - m[k]) ** 2 * n).sum() / n.sum() / len(groups))
print("lam:", L); print("MAE CV:", m.round(2)); print(f"nhỏ nhất ở lam={L[k]}, SE={se:.2f}, ngưỡng={m[k] + se:.2f} -> lam 1-SE = {max(l for l, x in zip(L, m) if x <= m[k] + se)}")
