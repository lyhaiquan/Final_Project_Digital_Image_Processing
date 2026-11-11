"""Tác động xuống gram và kcal trên TEST: nhãn YOLO (conf 0.25) so với DINO + hạ ngưỡng.
Mô hình gram = models/gram_model.json trong repo (fit trên train+val), cơm = khẩu phần TB (không bật độ sâu).
Tổng khay tính trên MỌI món dự đoán, kể cả món "ma" (chỉ số cũ bỏ qua món ma).
"""
import sys, json, numpy as np, pandas as pd
from common import *
import eval_cls as E, postproc as PP
sys.path.insert(0, ROOT); import grams as G

REPO = "D:/CVision/Final_Project_Digital_Image_Processing/models"
GM = json.load(open(f"{REPO}/gram_model.json", encoding="utf-8"))
cal = pd.read_csv(f"{REPO}/calories.csv"); KC = dict(zip(cal["class"], cal.kcal_per_100g.fillna(0.0)))
pl = pd.read_csv(f"{ROOT}/grams_out/plate_areas.csv").set_index("image")
it = G.load_items(); te = it[it.split == "test"]
LAB = {im: {NAMES[int(l.split()[0])] for l in open(f"{D}/labels/test/{im[:-4]}.txt") if l.strip()} for im in te.image.unique()}
WT = {im: set(g) for im, g in te.groupby("image")["class"]}
E.key = "dinov2b224"
CFG = dict(w=0.8, t_low=0.1, nms=0.7, gate=0.5)


def ridge_area(lam=3.0):
    """log g = mu + a_món + b·log A, phạt lam·a_món² (gram2.py). Trả về cùng định dạng với GM["area"]."""
    import depth_eval_pred as DP
    f = DP.it.merge(DP.F[["image", "class", "area_px"]], on=["image", "class"], how="left")
    f["A"] = f.area_px / f.image.map(DP.pl.plate_px)
    f = f[f.area_px.notna() & (f["class"] != "rice") & f.A.notna() & (f.split != "test")]
    cls = sorted(set(f["class"])); X = np.zeros((len(f), 2 + len(cls))); X[:, 0] = 1; X[:, 1] = np.log(f.A.values)
    for i, c in enumerate(f["class"]): X[i, 2 + cls.index(c)] = 1
    y = np.log(f.grams.values); pen = np.r_[0, 0, np.full(len(cls), lam)]
    coef = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    return dict(intercept={"_other": float(coef[0]), **{c: float(coef[0] + coef[2 + i]) for i, c in enumerate(cls)}},
                slope=float(coef[1]), smearing=float(np.mean(np.exp(y - X @ coef))), ridge_lambda=lam)


AREA = GM["area"]       # đổi sang ridge_area() để thử hệ số mới
RICE_V = {}             # {ảnh: V} nếu muốn cơm theo độ sâu


def grams(cls, A, im=None):
    if cls == "rice":
        v = RICE_V.get(im)
        if v is not None and v > 0:
            r = GM["rice_depth"]; return float(np.exp(r["intercept"]["rice"] + r["slope"] * np.log(v)) * r["smearing"])
        return GM["rice_mean_g"]
    if A is None or not np.isfinite(A): return GM["class_mean_g"].get(cls, GM["global_mean_g"])
    a = AREA; b = a["intercept"].get(cls, a["intercept"]["_other"])
    return float(np.exp(b + a["slope"] * np.log(max(A, 1e-6))) * a["smearing"])


def predict(tag, dec):
    """dec[im] = (lớp [N], giữ [N]) -> bảng (ảnh, món, gram, kcal)"""
    rows = []
    for im in sorted(te.image.unique()):
        m, _, _ = load_pred(tag, "test", im); c, k = dec[im]
        H, W = m.shape[1:]; pf = pl.plate_px[im] / (pl.H[im] * pl.W[im])     # đĩa / ảnh (tỉ lệ, không phụ thuộc độ phân giải)
        for ci in sorted(set(c[k].tolist())):
            u = m[k & (c == ci)].any(0).sum() / (H * W)
            g = grams(NAMES[ci], u / pf if np.isfinite(pf) else None, im)
            rows.append(dict(image=im, cls=NAMES[ci], p=g, pk=g * KC.get(NAMES[ci], 0) / 100))
    return pd.DataFrame(rows)


def metrics(P):
    t = te.rename(columns={"class": "cls"})[["image", "cls", "grams"]]
    # món CÓ trong nhãn nhưng không được cân (vd cơm của 2 khay test) -> bỏ qua, không coi là món "ma"
    P = P[[(c in LAB[im]) <= (c in WT[im]) for im, c in zip(P.image, P.cls)]]
    d = t.merge(P, on=["image", "cls"], how="outer")
    hit = d.grams.notna() & d.p.notna(); miss = d.grams.notna() & d.p.isna(); ghost = d.grams.isna()
    e = (d.p - d.grams).abs()[hit]
    d["tk"] = d.grams.fillna(0) * d.cls.map(KC).fillna(0) / 100
    tr = d.fillna(0).groupby("image").agg(g=("grams", "sum"), p=("p", "sum"), tk=("tk", "sum"), pk=("pk", "sum"))
    old = d[~ghost].fillna(0).groupby("image").agg(g=("grams", "sum"), p=("p", "sum"))
    return dict(cap=int((~ghost).sum()), sot=int(miss.sum()), ma=int(ghost.sum()), MAE_mon=e.mean(), medAPE_mon=np.median(e / d.grams[hit]) * 100,
                khay_g_MAE=(tr.p - tr.g).abs().mean(), khay_g_medAPE=np.median((tr.p - tr.g).abs() / tr.g) * 100, khay_g_lech=(tr.p - tr.g).mean(),
                khay_g_MAE_cu=(old.p - old.g).abs().mean(),
                khay_kcal_MAE=(tr.pk - tr.tk).abs().mean(), khay_kcal_medAPE=np.median((tr.pk - tr.tk).abs() / tr.tk) * 100, khay_kcal_lech=(tr.pk - tr.tk).mean())


if __name__ == "__main__":
    import depth_eval_pred as DP
    tags = ("full0", "full1", "full2", "fullE")
    _, Ps = E.run("test", tags, ("train", "val"), ("mp", "lr", 100.0))
    v = DP.F[DP.F["class"] == "rice"]; v = dict(zip(v.image, v.hsum_spread / v.image.map(pl.plate_px)))
    OLD, NEW = GM["area"], ridge_area(3.0)
    rows = []
    for t in tags:
        cd = covdata(t, "test"); pi = pairiou(t, "test"); ims = E.L(f"pred_{t}")["im"]
        base = {im: (cd[im][2], cd[im][3] >= 0.25) for im in cd}
        new = {im: PP.decide(cd[im][2], cd[im][3], Ps[t][ims == im], pi[im][0], **CFG)[:2] for im in cd}
        for rname, rv in (("khẩu phần TB", {}), ("độ sâu", v)):
            for dname, dec in (("YOLO", base), ("DINO+hạ ngưỡng", new)):
                for gname, ar in (("cũ", OLD), ("ridge", NEW)):
                    globals()["AREA"], globals()["RICE_V"] = ar, rv
                    rows.append(dict(model=t, com=rname, nhan_dien=dname, he_so_gram=gname, **metrics(predict(t, dec))))
    d = pd.DataFrame(rows); pd.set_option("display.width", 300)
    cols = ["sot", "ma", "MAE_mon", "medAPE_mon", "khay_g_MAE", "khay_g_medAPE", "khay_g_lech", "khay_kcal_MAE", "khay_kcal_medAPE", "khay_kcal_lech"]
    print("model đang dùng (full1):"); print(d[d.model == "full1"].drop(columns=["model", "cap", "khay_g_MAE_cu"]).round(1).to_string(index=False))
    print(); print("Trung bình 3 seed:"); print(d[d.model != "fullE"].groupby(["com", "nhan_dien", "he_so_gram"], sort=False)[cols].mean().round(1).to_string())
    print(); print("Gộp 3 seed (fullE):"); print(d[d.model == "fullE"].drop(columns=["model", "cap", "khay_g_MAE_cu"]).round(1).to_string(index=False))
    d.to_csv(f"{ROOT}/improve/result_downstream.csv", index=False)
