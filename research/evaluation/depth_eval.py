"""Depth-Anything-V2 có giúp ước lượng gram không? Phân tích đặc trưng từ kaggle/out_depth/depth_features.csv

Đặc trưng mỗi (ảnh, món), mask nhãn tay:
  A   = diện tích món / diện tích đĩa                    (như grams.py)
  V_k = tổng độ cao dương so với mặt đĩa / diện tích đĩa  (k = cách chuẩn hoá thang độ sâu: raw / dmed / spread)
So sánh (CV để riêng 1 nhóm buổi chụp, trên train+val; test chỉ in ra, không dùng để chọn):
  area   : log g = a_món + b·log A                    (cách đang dùng)
  vol    : log g = a_món + b·log V
  both   : log g = a_món + b1·log A + b2·log V
Riêng cơm: tương quan Spearman giữa gram và A / V (theo khay), và MAE cơm của từng cách so với "khẩu phần TB".

    python depth_eval.py
"""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:/food-labeling")
import grams as G

F = pd.read_csv(r"D:/food-labeling/kaggle/out_depth/depth_features.csv")
F = F[F.plane_ok == True]
pl = pd.read_csv(f"{G.OUT}/plate_areas.csv").set_index("image")
it = G.load_items()
MIN_TRAYS = 3


def design(d, feats, lev):
    X = np.zeros((len(d), len(lev) + len(feats)))
    for i, c in enumerate(d["cls_"]): X[i, lev.index(c)] = 1
    for j, f in enumerate(feats): X[:, len(lev) + j] = np.log(np.clip(d[f].values, 1e-6, None))
    return X


def fit_predict(tr, te, feats):
    ok = tr.groupby("class").tray.nunique(); ok = set(ok[ok >= MIN_TRAYS].index)
    tr = tr.assign(cls_=[c if c in ok else "_other" for c in tr["class"]])
    te = te.assign(cls_=[c if c in ok else "_other" for c in te["class"]])
    lev = sorted(set(tr.cls_) | {"_other"})
    X = design(tr, feats, lev); y = np.log(tr.grams.values)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if "_other" not in set(tr.cls_): coef[lev.index("_other")] = np.median(coef[:len(lev)][[lev.index(c) for c in set(tr.cls_)]])
    s = np.mean(np.exp(y - X @ coef))                     # Duan smearing
    return np.exp(design(te, feats, lev) @ coef) * s, coef[len(lev):]


def main():
    out = []
    for m in sorted(F.model.unique()):
        f = F[F.model == m].merge(it, on=["image", "class"])
        f["A"] = f.area_px / f.image.map(pl.plate_px)
        for k in ("raw", "dmed", "spread"):
            f[f"V_{k}"] = f[f"hsum_{k}"] / f.image.map(pl.plate_px)
        fs, te = f[f.split != "test"], f[f.split == "test"]

        # 1) cơm: tương quan theo khay (trung bình các ảnh cùng khay)
        r = f[f["class"] == "rice"].groupby("tray").mean(numeric_only=True)
        sp = {c: r.grams.rank().corr(r[c].rank()) for c in ["A", "V_raw", "V_dmed", "V_spread"]}
        print(f"\n=== model {m} | cơm: Spearman(gram, ·) trên {len(r)} khay:", {k: round(v, 2) for k, v in sp.items()})

        variants = {"area": ["A"]}
        for k in ("raw", "dmed", "spread"):
            variants[f"vol_{k}"] = [f"V_{k}"]; variants[f"both_{k}"] = ["A", f"V_{k}"]
        for name, feats in variants.items():
            cv = []
            for g in sorted(fs.group.unique()):
                tr, va = fs[fs.group != g], fs[fs.group == g]
                p, _ = fit_predict(tr, va, feats); cv.append(va.assign(p=p))
            cv = pd.concat(cv)
            mean_rice = []   # mốc cơm: khẩu phần TB (cùng CV)
            for g in sorted(fs.group.unique()):
                tr, va = fs[fs.group != g], fs[fs.group == g]
                mean_rice.append(va[va["class"] == "rice"].assign(p=tr[tr["class"] == "rice"].grams.mean()))
            mr = pd.concat(mean_rice)
            pt, b = fit_predict(fs, te, feats); te_ = te.assign(p=pt)
            rc, nr = cv[cv["class"] == "rice"], cv[cv["class"] != "rice"]
            out.append(dict(model=m, cach=name, he_so=np.round(b, 2).tolist(),
                            cv_MAE=(cv.p - cv.grams).abs().mean(), cv_MAE_tru_com=(nr.p - nr.grams).abs().mean(),
                            cv_MAE_com=(rc.p - rc.grams).abs().mean(), cv_MAE_com_TB=(mr.p - mr.grams).abs().mean(),
                            test_MAE=(te_.p - te_.grams).abs().mean(),
                            test_MAE_tru_com=(te_[te_["class"] != "rice"].p - te_[te_["class"] != "rice"].grams).abs().mean(),
                            test_MAE_com=(te_[te_["class"] == "rice"].p - te_[te_["class"] == "rice"].grams).abs().mean()))
    R = pd.DataFrame(out); pd.set_option("display.width", 250)
    print("\n" + R.round(1).to_string(index=False))
    R.to_csv(r"D:/food-labeling/grams_out/depth_eval.csv", index=False)


if __name__ == "__main__":
    main()
