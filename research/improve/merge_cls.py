"""Thử nghiệm 7: gộp các món nhìn gần giống nhau thành một nhóm (không train lại).

Nhóm được chọn TRƯỚC khi nhìn test, theo quy tắc trên val (conf_val.py): nhầm >= 3 lần/seed và kcal/100g chênh <= 20%.
  A: lạp xưởng+xúc xích, gà rán+gà chua ngọt, cá viên+trứng cút, su su+rau xào thập cẩm
  B: A + thịt kho+đậu sốt thịt (nhầm nhiều nhất nhưng kcal chênh 55%)
Cách gộp: cộng xác suất DINO và "xác suất" YOLO trong nhóm rồi kết hợp như cũ; ngưỡng giữ nguyên (0.1 / 0.7 / 0.5 / cờ 0.75).
Gram: fit lại log-log trên mức nhóm (cùng quy tắc MIN_TRAYS, smearing). kcal/100g của nhóm = TB theo gram của các thành viên.
Sự thật: gram và kcal của TỪNG thành viên theo kcal riêng của nó (không quy về kcal nhóm).

    python merge_cls.py
"""
import sys, collections, numpy as np, pandas as pd
from common import *
import eval_cls as E
sys.path.insert(0, ROOT)
import depth_eval as DE, depth_eval_pred as DP, grams as G

E.key = "dinov2b224"
W8, T_LOW, NMS, GATE, T_HI, FLAG = 0.8, 0.1, 0.7, 0.5, 0.25, 0.75
REPO = "D:/CVision/Final_Project_Digital_Image_Processing/models"
cal = pd.read_csv(f"{REPO}/calories.csv"); KC = dict(zip(cal["class"], cal.kcal_per_100g.fillna(0.0)))
GM = __import__("json").load(open(f"{REPO}/gram_model.json", encoding="utf-8"))
pl = pd.read_csv(f"{ROOT}/grams_out/plate_areas.csv").set_index("image")
IT = G.load_items()

A = [("chinese_sausage", "sausage"), ("fried_chicken", "sweet_sour_chicken"), ("fish_ball", "quail_egg"), ("chayote", "stir_fried_vegetables")]
VARIANTS = {"không gộp": [], "A (4 cặp, kcal gần)": A, "B (A + thịt kho/đậu sốt thịt)": A + [("coconut_braised_pork", "pork_with_tofu")]}


def grouping(pairs):
    gid = {n: n for n in NAMES}
    for a, b in pairs: gid[b] = gid[a] = "+".join((a, b))
    gnames = sorted(set(gid.values())); M = np.zeros((K, len(gnames)))
    for i, n in enumerate(NAMES): M[i, gnames.index(gid[n])] = 1
    return gid, gnames, M


def fit_gram(gid, fit_splits):
    """log g = a_nhóm + b·log A trên mask dự đoán (như export_gram_model), dữ liệu gộp theo (ảnh, nhóm)."""
    f = DP.it.merge(DP.F[["image", "class", "area_px"]], on=["image", "class"], how="left")
    f = f[f.split.isin(fit_splits) & f.area_px.notna()].assign(g=lambda d: d["class"].map(gid))
    f = f.groupby(["image", "g"]).agg(grams=("grams", "sum"), area_px=("area_px", "sum"), tray=("tray", "first")).reset_index()
    f["A"] = f.area_px / f.image.map(pl.plate_px); f = f[f.A.notna()].rename(columns={"g": "class"})
    ok = f.groupby("class").tray.nunique(); ok = set(ok[ok >= DE.MIN_TRAYS].index)
    cls = [c if c in ok else "_other" for c in f["class"]]; lev = sorted(set(cls) | {"_other"})
    X = np.zeros((len(f), len(lev) + 1)); X[np.arange(len(f)), [lev.index(c) for c in cls]] = 1; X[:, -1] = np.log(f.A.values)
    y = np.log(f.grams.values); coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(intercept=dict(zip(lev, coef[:-1])), slope=coef[-1], smearing=float(np.mean(np.exp(y - X @ coef))))


def group_kcal(gid, fit_splits):
    w = IT[IT.split.isin(fit_splits)].assign(g=lambda d: d["class"].map(gid), k=lambda d: d.grams * d["class"].map(KC) / 100)
    s = w.groupby("g").agg(k=("k", "sum"), g_=("grams", "sum"))
    out = {g: KC[g] for g in set(gid.values()) if g in KC}
    out.update({g: 100 * r.k / r.g_ for g, r in s.iterrows() if "+" in g})
    for g in set(gid.values()):
        if g not in out: out[g] = np.mean([KC[n] for n in g.split("+")])
    return out


def decide(c, s, P, iou, M):
    n = len(c)
    if n == 0: return np.zeros(0, int), np.zeros(0, bool), np.zeros(0)
    py = np.repeat(((1 - s) / (K - 1))[:, None], K, 1); py[np.arange(n), c] = s
    lp = W8 * np.log(P @ M + 1e-6) + (1 - W8) * np.log(py @ M + 1e-6)
    q = np.exp(lp - lp.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
    cls, pm = q.argmax(1), q.max(1)
    keep = (s >= T_LOW) & ((s >= T_HI) | (pm >= GATE))
    for i in np.argsort(-s, kind="stable"):
        if keep[i]:
            dup = (iou[i] >= NMS) & ((s < s[i]) | ((s == s[i]) & (np.arange(n) > i))); keep[dup] = False
    return cls, keep, pm


def evaluate(split, tags, ref, fit_splits, pairs, coarse_eval=None):
    """coarse_eval: chấm nhận diện ở mức nhóm này (để so công bằng, kể cả khi không gộp lúc dự đoán)."""
    gid, gn, M = grouping(pairs); gidx = {g: i for i, g in enumerate(gn)}
    eg = coarse_eval or gid
    gm, kc = fit_gram(gid, fit_splits), group_kcal(gid, fit_splits)
    _, Ps = E.run(split, tags, ref, ("mp", "lr", 100.0))
    it = IT[IT.split == split]
    LAB = {im: {NAMES[int(l.split()[0])] for l in open(f"{D}/labels/{split}/{im[:-4]}.txt") if l.strip()} for im in images(split)}
    out = []
    for t in tags:
        cd = covdata(t, split); pi = pairiou(t, split); ims = E.L(f"pred_{t}")["im"]
        ok = wrong = miss = 0; items = []; trays = []
        for im, (g, cv, c, s) in cd.items():
            cls, keep, pm = decide(c, s, Ps[t][ims == im], pi[im][0], M)
            m, _, _ = load_pred(t, split, im)
            # nhận diện (mức nhóm eg)
            pg = np.array([eg[gn[x].split("+")[0]] for x in cls]) if len(cls) else np.array([])
            for gi, row in zip(g, cv):
                kk = keep.copy(); best, f = None, 0.0
                for x in set(pg[kk].tolist()):
                    v = row[kk & (pg == x)].max()
                    if v > f: best, f = x, v
                pred = best if f >= 0.3 else None
                ok += pred == eg[NAMES[gi]]; miss += pred is None; wrong += pred not in (eg[NAMES[gi]], None)
            # món hiển thị (mức nhóm dự đoán gid)
            H, W = m.shape[1:]; pf = pl.plate_px[im] / (pl.H[im] * pl.W[im]) if im in pl.index else np.nan
            labg = {gid[n] for n in LAB[im]}
            for x in sorted(set(cls[keep].tolist())):
                k = keep & (cls == x); name = gn[x]
                A_ = m[k].any(0).sum() / (H * W) / pf if np.isfinite(pf) else np.nan
                if name == "rice": gram = GM["rice_mean_g"]
                elif np.isfinite(A_):
                    a = gm["intercept"].get(name, gm["intercept"]["_other"]); gram = float(np.exp(a + gm["slope"] * np.log(max(A_, 1e-6))) * gm["smearing"])
                else: gram = GM["global_mean_g"]
                items.append(dict(image=im, g=name, p=gram, pk=gram * kc[name] / 100, conf=pm[k].max(), err=name not in labg,
                                  err_c=eg[name.split("+")[0]] not in {eg[n] for n in LAB[im]}))
        P = pd.DataFrame(items)
        # gram / kcal trên ảnh có cân; thật theo TỪNG thành viên
        tr = it.assign(g=it["class"].map(gid), k=it.grams * it["class"].map(KC) / 100).groupby(["image", "g"]).agg(grams=("grams", "sum"), k=("k", "sum")).reset_index()
        WTg = {im: set(x.g) for im, x in tr.groupby("image")}
        Pw = P[P.image.isin(WTg)]
        Pw = Pw[[(name not in {gid[n] for n in LAB[im]}) or (name in WTg[im]) for im, name in zip(Pw.image, Pw.g)]]
        d = tr.merge(Pw, on=["image", "g"], how="outer"); hit = d.grams.notna() & d.p.notna()
        ty = d.fillna(0).groupby("image").agg(g=("grams", "sum"), p=("p", "sum"), k=("k", "sum"), pk=("pk", "sum"))
        n = ok + wrong + miss
        out.append(dict(model=t, vung_dung=ok, nham=wrong, sot=miss, acc=100 * ok / n, mon_hien=len(P), ma=int(P.err.sum()), ma_nhom=int(P.drop_duplicates(["image"]).shape[0] * 0 + P.assign(k=[eg[x.split("+")[0]] for x in P.g]).drop_duplicates(["image", "k"]).err_c.sum()),
                        loi_lot_nhom=int(P.assign(k=[eg[x.split("+")[0]] for x in P.g]).sort_values("conf", ascending=False).drop_duplicates(["image", "k"]).pipe(lambda q: (q.err_c & (q.conf >= FLAG)).sum())),
                        co_xac_nhan=100 * (P.conf < FLAG).mean(), loi_lot=int((P.err & (P.conf >= FLAG)).sum()),
                        sot_can=int((d.grams.notna() & d.p.isna()).sum()), MAE_mon=(d.p - d.grams).abs()[hit].mean(),
                        khay_g_MAE=(ty.p - ty.g).abs().mean(), khay_kcal_MAE=(ty.pk - ty.k).abs().mean(),
                        khay_kcal_medAPE=np.median((ty.pk - ty.k).abs() / ty.k) * 100, khay_kcal_lech=(ty.pk - ty.k).mean()))
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    # kiểm tra: không gộp + fit trên train+val phải tái tạo gram_model.json
    g0 = fit_gram(grouping([])[0], ("train", "val"))
    print("kiểm tra hệ số gram: slope", round(g0["slope"], 4), "vs", round(GM["area"]["slope"], 4), "| smearing", round(g0["smearing"], 4), "vs", round(GM["area"]["smearing"], 4))
    for split, tags, ref, fs in (("val", ("tr0", "tr1", "tr2"), ("train",), ("train",)), ("test", ("full0", "full1", "full2"), ("train", "val"), ("train", "val"))):
        rows = []
        for vn, pairs in VARIANTS.items():
            r = evaluate(split, tags, ref, fs, pairs).assign(cach=vn); rows.append(r)
            if pairs:   # so công bằng: pipeline KHÔNG gộp nhưng chấm nhận diện ở cùng mức nhóm
                rows.append(evaluate(split, tags, ref, fs, [], coarse_eval=grouping(pairs)[0]).assign(cach=f"không gộp, chấm theo nhóm {vn[0]}"))
        d = pd.concat(rows)
        cols = ["vung_dung", "nham", "sot", "acc", "ma", "ma_nhom", "co_xac_nhan", "loi_lot", "loi_lot_nhom", "sot_can", "MAE_mon", "khay_g_MAE", "khay_kcal_MAE", "khay_kcal_medAPE", "khay_kcal_lech"]
        print(f"\n===== {split.upper()} (TB 3 seed; val: chỉ 15 ảnh có cân, gram fit trên train)")
        print(d.groupby("cach", sort=False)[cols].mean().round(1).to_string())
        print("theo từng model, kcal khay MAE:")
        print(d.pivot_table(index="cach", columns="model", values="khay_kcal_MAE", sort=False).round(1).to_string())
        d.to_csv(f"{ROOT}/improve/result_merge_{split}.csv", index=False)
