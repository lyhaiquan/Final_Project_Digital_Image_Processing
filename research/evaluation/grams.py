"""Ước lượng gram từ diện tích mask (2D baseline).

Đơn vị đánh giá: (ảnh, lớp). Gram = tổng các món cùng lớp trong ảnh (585/585 cặp đã cân đủ mọi polygon).
Diện tích = diện tích hợp các mask của lớp / diện tích ảnh  (GT, PRED)
          hoặc / diện tích đĩa từ plate.py                (GT-dia, PRED-dia) -> khử khoảng cách & góc chụp

So 3 cách, mỗi cách trên 2 nguồn mask (GT = nhãn tay, PRED = model food_seg_v4_final.pt):
  mean   : đoán gram trung bình của lớp, KHÔNG nhìn ảnh  (mốc so sánh bắt buộc)
  ratio  : g = k_lớp * diện_tích                        (k = trung vị g/diện_tích)
  loglog : log g = a_lớp + b * log diện_tích            (b dùng chung, a riêng mỗi lớp)
  hybrid : cơm dùng "mean", món khác dùng "loglog"   (chọn theo CV: diện tích cơm không giúp vì cơm bị món khác đè lên)
Lớp có < MIN_TRAYS khay khác nhau trong tập fit -> dùng hệ số chung.

Chọn cách bằng CV theo nhóm buổi chụp trên train+val; test (5 nhóm) chỉ chấm MỘT lần ở cuối.

    python grams.py
"""
import os, sys, collections
import numpy as np, pandas as pd, cv2

sys.stdout.reconfigure(encoding="utf-8")
D = r"D:/food_dataset_v4"
MODEL = r"D:/food-labeling/models/food_seg_v4_final.pt"
OUT = r"D:/food-labeling/grams_out"
CONF, MIN_TRAYS = 0.25, 3
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]


def load_items():
    w = pd.read_csv(f"{D}/weights.csv").drop(columns="split")
    m = pd.read_csv(f"{D}/manifest.csv")[["image", "split", "group"]]
    w = w.merge(m, on="image")
    # khay = tập (lớp, gram, nguồn cân) giống hệt nhau -> cùng một khay chụp nhiều góc
    key = w.groupby("image").apply(lambda g: hash(tuple(sorted(zip(g["class"], g.grams, g.weight_source.astype(str))))),
                                   include_groups=False)
    w["tray"] = w.image.map(key).factorize()[0]
    w["box"] = w.weight_source.astype(str).str.contains("hộp")
    it = (w.groupby(["image", "class"])
            .agg(grams=("grams", "sum"), split=("split", "first"), group=("group", "first"),
                 tray=("tray", "first"), box=("box", "any"))
            .reset_index())
    return it


def gt_areas(images, split_of):
    out = {}
    for im in images:
        p = f"{D}/images/{split_of[im]}/{im}"
        H, W = cv2.imread(p).shape[:2]
        ms = {}
        for l in open(f"{D}/labels/{split_of[im]}/{im[:-4]}.txt"):
            r = l.split()
            if not r: continue
            c = NAMES[int(r[0])]
            xy = (np.array(r[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
            cv2.fillPoly(ms.setdefault(c, np.zeros((H, W), np.uint8)), [xy], 1)
        for c, mk in ms.items(): out[(im, c)] = mk.sum() / (H * W)
    return out


def pred_areas(images, split_of):
    cache = f"{OUT}/pred_areas.csv"
    if os.path.exists(cache):
        d = pd.read_csv(cache); return {(a, b): v for a, b, v in d.itertuples(index=False)}
    from ultralytics import YOLO
    model, out = YOLO(MODEL), {}
    for im in images:
        r = model.predict(f"{D}/images/{split_of[im]}/{im}", conf=CONF, retina_masks=True, verbose=False)[0]
        H, W = r.orig_shape; ms = {}
        if r.masks is not None:
            for mk, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int)):
                n = model.names[c]; ms[n] = ms.get(n, np.zeros((H, W), bool)) | mk
        for c, mk in ms.items(): out[(im, c)] = mk.sum() / (H * W)
    pd.DataFrame([(a, b, v) for (a, b), v in out.items()], columns=["image", "class", "area"]).to_csv(cache, index=False)
    return out


class Model:
    def __init__(self, kind): self.kind = kind

    def fit(self, d):
        if self.kind == "hybrid":   # cơm: khẩu phần trung bình (CV: diện tích cơm không giúp); món khác: loglog
            self.m = Model("mean").fit(d); self.l = Model("loglog").fit(d)
            # Duan smearing: exp(fit log) cho trung bình nhân -> đoán thấp có hệ thống, cộng dồn ra tổng khay bị hụt.
            # CV: sai lệch tổng khay -24 g -> -11 g, MAE khay 111 -> 107 g. Chỉ áp cho phần loglog (món khác cơm).
            t = d[(d.area > 0) & (d["class"] != "rice")]
            self.s = float(np.mean(np.exp(np.log(t.grams.values) - np.log(self.l.predict(t)))))
            return self
        d = d[d.area > 0]
        ok = d.groupby("class").tray.nunique(); ok = set(ok[ok >= MIN_TRAYS].index)
        self.ok = ok
        if self.kind == "mean":
            self.glob = d.grams.mean(); self.c = d[d["class"].isin(ok)].groupby("class").grams.mean().to_dict()
        elif self.kind == "ratio":
            r = d.grams / d.area
            self.glob = r.median(); self.c = r[d["class"].isin(ok)].groupby(d["class"]).median().to_dict()
        else:  # loglog: log g = a_c + b log A, b chung
            la, lg = np.log(d.area.values), np.log(d.grams.values)
            cls = [c if c in ok else "_other" for c in d["class"]]
            lev = sorted(set(cls)); X = np.zeros((len(d), len(lev) + 1))
            for i, c in enumerate(cls): X[i, lev.index(c)] = 1
            X[:, -1] = la
            coef, *_ = np.linalg.lstsq(X, lg, rcond=None)
            self.b = coef[-1]; self.c = dict(zip(lev, coef[:-1])); self.glob = self.c.get("_other", np.median(coef[:-1]))
            # nếu không có lớp hiếm nào trong tập fit: hệ số chung = intercept trung vị
        return self

    def predict(self, d):
        if self.kind == "hybrid":
            return np.where(d["class"].values == "rice", self.m.predict(d), self.s * self.l.predict(d))
        if self.kind == "mean":
            return np.array([self.c.get(c, self.glob) for c in d["class"]])
        if self.kind == "ratio":
            return np.array([self.c.get(c, self.glob) * a for c, a in zip(d["class"], d.area)])
        return np.array([np.exp(self.c.get(c, self.glob) + self.b * np.log(max(a, 1e-6))) for c, a in zip(d["class"], d.area)])


def metrics(d, p):
    hit = d.area.values > 0
    e = np.abs(p - d.grams.values)
    tray = pd.DataFrame({"image": d.image.values, "t": d.grams.values, "p": np.where(hit, p, 0)}).groupby("image").sum()
    return dict(n=len(d), miss=int((~hit).sum()), MAE_g=e[hit].mean(), medAPE=np.median(e[hit] / d.grams.values[hit]) * 100,
                tray_MAE_g=np.abs(tray.p - tray.t).mean(), tray_medAPE=np.median(np.abs(tray.p - tray.t) / tray.t) * 100)


def main():
    os.makedirs(OUT, exist_ok=True)
    it = load_items()
    split_of = dict(zip(it.image, it.split)); imgs = sorted(split_of)
    print(f"{len(it)} cặp (ảnh,lớp) | {it.image.nunique()} ảnh | {it.tray.nunique()} khay khác nhau | "
          f"test: {(it.split == 'test').sum()} cặp, {it[it.split == 'test'].tray.nunique()} khay")
    A = {"GT": gt_areas(imgs, split_of), "PRED": pred_areas(imgs, split_of)}
    # chuẩn hoá theo ĐĨA: diện tích món / diện tích đĩa (plate.py). Ảnh không thấy đĩa -> tỉ lệ đĩa/ảnh trung vị
    pl = pd.read_csv(f"{OUT}/plate_areas.csv"); pf = (pl.plate_px / (pl.H * pl.W))
    pf = dict(zip(pl.image, pf.fillna(pf.median())))
    for src in ("GT", "PRED"):
        A[src + "-dia"] = {k: v / pf[k[0]] for k, v in A[src].items()}

    fitset = it[it.split != "test"]; test = it[it.split == "test"]
    groups = sorted(fitset.group.unique())
    rows, preds = [], []
    for src, area in A.items():
        for kind in ("mean", "ratio", "loglog", "hybrid"):
            # CV theo nhóm (leave-one-group-out) trên train+val
            cv = []
            for g in groups:
                tr = fitset[fitset.group != g].assign(area=lambda x: [area.get(k, 0) for k in zip(x.image, x["class"])])
                va = fitset[fitset.group == g].assign(area=lambda x: [area.get(k, 0) for k in zip(x.image, x["class"])])
                m = Model(kind).fit(tr); cv.append(va.assign(p=m.predict(va)))
            cv = pd.concat(cv)
            mc = metrics(cv, cv.p.values)
            tr = fitset.assign(area=[area.get(k, 0) for k in zip(fitset.image, fitset["class"])])
            te = test.assign(area=[area.get(k, 0) for k in zip(test.image, test["class"])])
            m = Model(kind).fit(tr); p = m.predict(te); mt = metrics(te, p)
            rows.append(dict(mask=src, method=kind, **{f"cv_{k}": v for k, v in mc.items()}, **{f"test_{k}": v for k, v in mt.items()}))
            preds.append(te.assign(mask=src, method=kind, pred=p))
    R = pd.DataFrame(rows); P = pd.concat(preds)
    R.to_csv(f"{OUT}/summary.csv", index=False); P.to_csv(f"{OUT}/test_predictions.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(1).to_string(index=False))

    # theo lớp + cơm riêng + hộp vs đĩa, cho cách tốt nhất theo CV trên mỗi nguồn mask
    for src in A:
        best = R[R["mask"] == src].sort_values("cv_MAE_g").iloc[0].method
        q = P[(P["mask"] == src) & (P.method == best) & (P.area > 0)].copy()
        q["err"] = (q.pred - q.grams).abs(); q["ape"] = q.err / q.grams * 100
        print(f"\n== {src}, cách {best}: theo lớp (test)")
        print(q.groupby("class").agg(n=("err", "size"), g_that=("grams", "mean"), MAE=("err", "mean"), medAPE=("ape", "median"))
              .round(1).sort_values("n", ascending=False).to_string())
        print("hộp vs đĩa:", q.groupby("box").agg(n=("err", "size"), MAE=("err", "mean"), medAPE=("ape", "median")).round(1).to_dict("index"))


if __name__ == "__main__":
    main()
