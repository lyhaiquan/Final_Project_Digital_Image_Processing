"""Hướng 5: "±30%" phủ được bao nhiêu? Khoảng sai số trung thực từ phần dư CV (chỉ món nhận đúng; mô hình gram hiện tại)."""
import sys, numpy as np, pandas as pd
from common import *
sys.path.insert(0, ROOT)
import depth_eval_pred as DP
f = DP.it.merge(DP.F[["image", "class", "area_px", "hsum_spread"]], on=["image", "class"], how="left")
f["hit"] = f.area_px.notna(); f["A"] = f.area_px / f.image.map(DP.pl.plate_px); f["V"] = f.hsum_spread / f.image.map(DP.pl.plate_px)
fs, te = f[f.split != "test"], f[f.split == "test"]
rows = []
for rv in (False, True):
    cv = pd.concat([fs[fs.group == g].assign(p=DP.hyb(fs[fs.group != g], fs[fs.group == g], rv)) for g in sorted(fs.group.unique())]); t = te.assign(p=DP.hyb(fs, te, rv))
    cv, t = cv[cv.hit & cv.p.notna()], t[t.hit & t.p.notna()]
    for name, sel in (("món khác cơm", lambda d: d[d["class"] != "rice"]), ("cơm", lambda d: d[d["class"] == "rice"])):
        if rv != (name == "cơm") and not (not rv and name == "cơm"): continue
        c, x = sel(cv), sel(t); rc, rx = (c.p / c.grams).values, (x.p / x.grams).values
        lo, hi = np.quantile(c.grams / c.p, [0.1, 0.9])      # thật / đoán: khoảng 80% từ CV
        rows.append(dict(nhom=name + (" (độ sâu)" if rv and name == "cơm" else " (khẩu phần TB)" if name == "cơm" else ""), n_cv=len(c), n_test=len(x),
                         phu_30_cv=round(100 * (np.abs(c.grams / c.p - 1) <= 0.3).mean()), phu_30_test=round(100 * (np.abs(x.grams / x.p - 1) <= 0.3).mean()), phu_50_cv=round(100 * (np.abs(c.grams / c.p - 1) <= 0.5).mean()), phu_50_test=round(100 * (np.abs(x.grams / x.p - 1) <= 0.5).mean()),
                         khoang_80=f"{100 * (lo - 1):+.0f}% .. {100 * (hi - 1):+.0f}%", phu_khoang_80_test=round(100 * (((x.grams / x.p) >= lo) & ((x.grams / x.p) <= hi)).mean())))
pd.set_option("display.width", 200); print(pd.DataFrame(rows).to_string(index=False))
