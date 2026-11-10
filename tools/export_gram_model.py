"""Fit công thức gram MỘT lần trên train+val rồi lưu ra models/gram_model.json (để predict.py dùng, không cần dữ liệu cân).

Cần bộ dữ liệu gốc (không có trong repo): D:/food_dataset_v4, D:/food-labeling/{grams_out,kaggle/out_depth_pred}.
Logic giống hệt research/depth_eval.py::fit_predict (log g = a_món + b·log x, lớp < 3 khay -> "_other", Duan smearing).
Kiểm tra: chấm lại test phải ra đúng Bảng V của báo cáo (MAE món 26.8 g / 23.8 g, MAE tổng khay 112.9 g / 82.8 g).

    python tools/export_gram_model.py
"""
import json, os, sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:/food-labeling")
import depth_eval as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "models", "gram_model.json")


def fit(tr, feat):
    """Như E.fit_predict nhưng trả về tham số."""
    ok = tr.groupby("class").tray.nunique(); ok = set(ok[ok >= E.MIN_TRAYS].index)
    tr = tr.assign(cls_=[c if c in ok else "_other" for c in tr["class"]])
    lev = sorted(set(tr.cls_) | {"_other"})
    X = E.design(tr, [feat], lev); y = np.log(tr.grams.values)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if "_other" not in set(tr.cls_):
        coef[lev.index("_other")] = np.median(coef[:len(lev)][[lev.index(c) for c in set(tr.cls_)]])
    s = float(np.mean(np.exp(y - X @ coef)))
    return {"intercept": {c: float(coef[i]) for i, c in enumerate(lev)}, "slope": float(coef[len(lev)]), "smearing": s}


def predict_one(m, cls, x):
    a = m["intercept"][cls] if cls in m["intercept"] else m["intercept"]["_other"]
    return float(np.exp(a + m["slope"] * np.log(max(x, 1e-6))) * m["smearing"])


def grams(model, cls, A, V=None):
    """Hàm dùng chung với predict.py: cơm = thể tích độ sâu (nếu có) hoặc khẩu phần TB; món khác = diện tích/đĩa."""
    if cls == "rice":
        return predict_one(model["rice_depth"], "rice", V) if (V is not None and V > 0) else model["rice_mean_g"]
    return predict_one(model["area"], cls, A)


def main():
    F = pd.read_csv(r"D:/food-labeling/kaggle/out_depth_pred/depth_features_pred.csv")
    F = F[(F.plane_ok == True) & (F.model == "L")]
    f = E.it.merge(F[["image", "class", "area_px", "hsum_spread"]], on=["image", "class"], how="left")
    f["hit"] = f.area_px.notna()
    f["A"] = f.area_px / f.image.map(E.pl.plate_px); f["V"] = f.hsum_spread / f.image.map(E.pl.plate_px)
    fs = f[(f.split != "test") & f.hit]; te = f[f.split == "test"]

    model = {
        "description": "Hệ số ước lượng gram, fit trên train+val (mask dự đoán của food_seg_v4_final.pt). "
                       "A = diện tích mask món / diện tích đĩa; V = tổng độ cao dương (Depth-Anything-V2 Large) / diện tích đĩa.",
        "area": fit(fs, "A"),
        "rice_depth": fit(fs, "V"),
        "rice_mean_g": float(fs[fs["class"] == "rice"].grams.mean()),
        # dự phòng khi không thấy đĩa: khẩu phần TB từng món (mốc "mean" của báo cáo)
        "class_mean_g": {c: float(v) for c, v in fs.groupby("class").grams.mean().items()},
        "global_mean_g": float(fs.grams.mean()),
        "min_trays": E.MIN_TRAYS,
    }
    # nhóm hay nhầm (từ bảng nhầm lẫn trên test, cặp xuất hiện >= 2 lần, bỏ cặp với cơm vì chủ yếu do che lấp)
    cf = pd.read_csv(r"D:/food-labeling/grams_out/confusion_test.csv")
    w = cf[(cf.that != cf.doan) & (cf.doan != "(bỏ sót)") & (cf.that != "rice") & (cf.doan != "rice")]
    conf = {}
    for (a, b), n in w.groupby(["that", "doan"]).size().items():
        if n >= 2:
            conf.setdefault(a, []).append(b); conf.setdefault(b, []).append(a)
    model["confusions"] = {k: sorted(set(v)) for k, v in sorted(conf.items())}
    model["rice_depth"] = {"intercept": {"rice": model["rice_depth"]["intercept"]["rice"]},
                           "slope": model["rice_depth"]["slope"], "smearing": model["rice_depth"]["smearing"]}

    # --- kiểm tra lại với Bảng V ---
    for use_depth in (False, True):
        t = te.copy()
        t["p"] = [grams(model, c, a, v if use_depth else None) if h else np.nan
                  for c, a, v, h in zip(t["class"], t.A, t.V, t.hit)]
        h = t[t.hit]; e = (h.p - h.grams).abs()
        tray = t.assign(p=t.p.fillna(0)).groupby("image").agg(p=("p", "sum"), g=("grams", "sum"))
        print(("độ sâu " if use_depth else "diện tích"), f"MAE món {e.mean():.1f} g | MAE khay {np.abs(tray.p - tray.g).mean():.1f} g")
    json.dump(model, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("đã lưu", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
