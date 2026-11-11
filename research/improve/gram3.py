"""Ridge vs hiện tại: lệch có hệ thống và sai số TỔNG các món khác cơm của mỗi ảnh (CV và test)."""
import sys, numpy as np, pandas as pd
from common import *
sys.path.insert(0, ROOT)
import depth_eval as DE, depth_eval_pred as DP, gram2 as G2
f = DP.it.merge(DP.F[["image", "class", "area_px"]], on=["image", "class"], how="left")
f["A"] = f.area_px / f.image.map(DP.pl.plate_px)
f = f[f.area_px.notna() & (f["class"] != "rice") & f.A.notna()]
fs, te = f[f.split != "test"], f[f.split == "test"]; groups = sorted(fs.group.unique())
rows = []
for name, fp in [("hiện tại", lambda a, b: DE.fit_predict(a, b, ["A"])[0])] + [(f"ridge {l:g}", lambda a, b, l=l: G2.fit_predict(a, b, l, False)) for l in (1, 3, 10, 30)]:
    cv = pd.concat([fs[fs.group == g].assign(p=fp(fs[fs.group != g], fs[fs.group == g])) for g in groups]); t = te.assign(p=fp(fs, te))
    r = dict(cach=name)
    for k, d in (("cv", cv), ("test", t)):
        s = d.groupby("image").agg(p=("p", "sum"), g=("grams", "sum"))
        r.update({f"{k}_MAE_mon": (d.p - d.grams).abs().mean(), f"{k}_lech_mon": (d.p - d.grams).mean(), f"{k}_MAE_tong_anh": (s.p - s.g).abs().mean(), f"{k}_lech_tong_anh": (s.p - s.g).mean()})
    rows.append(r)
pd.set_option("display.width", 250); print(pd.DataFrame(rows).round(1).to_string(index=False))
