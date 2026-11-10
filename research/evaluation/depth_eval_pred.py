"""Như depth_eval.py nhưng với mask DO MODEL DỰ ĐOÁN (kaggle/out_depth_pred/depth_features_pred.csv).
Món model không nhận ra -> "miss": không tính vào MAE món, tính 0 g vào tổng khay (giống grams.py).
So 2 cách hybrid: cơm = khẩu phần TB  vs  cơm = thể tích độ sâu (Large, V_spread). Món khác: diện tích/đĩa.

    python depth_eval_pred.py
"""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:/food-labeling")
import depth_eval as E

F = pd.read_csv(r"D:/food-labeling/kaggle/out_depth_pred/depth_features_pred.csv")
F = F[(F.plane_ok == True) & (F.model == "L")]
pl, it = E.pl, E.it


def hyb(tr, va, rice_vol):
    trh, vah = tr[tr.hit], va[va.hit]
    p = np.full(len(va), np.nan)
    pa, _ = E.fit_predict(trh, vah, ["A"])
    if rice_vol: pr, _ = E.fit_predict(trh, vah, ["V"])
    else: pr = np.full(len(vah), trh[trh["class"] == "rice"].grams.mean())
    p[va.hit.values] = np.where(vah["class"].values == "rice", pr, pa)
    return p


def score(d):
    h = d[d.hit]; e = (h.p - h.grams).abs(); r = h["class"] == "rice"
    tray = d.assign(p=d.p.fillna(0)).groupby("image").agg(p=("p", "sum"), g=("grams", "sum"))
    te = np.abs(tray.p - tray.g)
    return dict(n=len(d), miss=int((~d.hit).sum()), MAE=e.mean(), MAE_com=e[r].mean(), MAE_tru_com=e[~r].mean(),
                khay_MAE=te.mean(), khay_medAPE=np.median(te / tray.g) * 100)


def main():
    f = it.merge(F[["image", "class", "area_px", "hsum_spread"]], on=["image", "class"], how="left")
    f["hit"] = f.area_px.notna()
    f["A"] = f.area_px / f.image.map(pl.plate_px); f["V"] = f.hsum_spread / f.image.map(pl.plate_px)
    fs, te = f[f.split != "test"], f[f.split == "test"]
    r = f[(f["class"] == "rice") & f.hit].groupby("tray").mean(numeric_only=True)
    print(f"cơm (mask dự đoán), Spearman trên {len(r)} khay: diện tích {r.grams.rank().corr(r.A.rank()):.2f} | thể tích {r.grams.rank().corr(r.V.rank()):.2f}")
    rows = []
    for rv in (False, True):
        cv = pd.concat([fs[fs.group == g].assign(p=hyb(fs[fs.group != g], fs[fs.group == g], rv)) for g in sorted(fs.group.unique())])
        rows.append(dict(com="độ sâu" if rv else "khẩu phần TB", **{f"cv_{k}": v for k, v in score(cv).items()},
                         **{f"test_{k}": v for k, v in score(te.assign(p=hyb(fs, te, rv))).items()}))
    R = pd.DataFrame(rows); pd.set_option("display.width", 250)
    print(R.round(1).T.to_string()); R.to_csv(r"D:/food-labeling/grams_out/depth_eval_pred.csv", index=False)


if __name__ == "__main__":
    main()
