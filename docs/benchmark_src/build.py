"""Sinh benchmark.html (khổ IEEE 2 cột) từ các CSV kết quả; mọi con số đều đọc từ file, không gõ tay.
    D:/food-labeling/.venv/Scripts/python.exe build.py
"""
import sys, glob, collections, html, numpy as np, pandas as pd, cv2
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:/food-labeling")
import depth_eval as E

K = r"D:/food-labeling/kaggle"
DS = r"D:/food_dataset_v4"
OUT = r"D:/CVision/benchmark_ieee"
VI = pd.read_csv(r"D:/food-labeling/calories_47.csv").set_index("class").ten_viet.to_dict()
VI.update({"coconut_braised_pork": "thịt kho", "pork_floss": "ruốc", "shrimp_spring_roll": "nem tôm",
           "sweet_sour_chicken": "gà chua ngọt", "rice": "cơm trắng"})


def ms(s, d=3):
    return f'{s.mean():.{d}f}<span class="sd">±{s.std(ddof=1):.{d}f}</span>'


# ---------- Bảng I: dữ liệu ----------
man = pd.read_csv(f"{DS}/manifest.csv")
w = pd.read_csv(f"{DS}/weights.csv")
inst = collections.Counter()
cls_split = collections.defaultdict(set)
for s in ("train", "val", "test"):
    for f in glob.glob(f"{DS}/labels/{s}/*.txt"):
        for l in open(f):
            if l.strip():
                inst[s] += 1; cls_split[s].add(int(l.split()[0]))
it = E.it
trays = it.groupby("split").tray.nunique()
pairs = it.groupby("split").size()
grp = man.groupby("split").group.nunique()
t1 = []
for s, name in (("train", "Train"), ("val", "Val"), ("test", "Test (khóa)")):
    t1.append([name, (man.split == s).sum(), grp[s], inst[s], len(cls_split[s]), pairs.get(s, 0), trays.get(s, 0)])
t1.append(["Tổng", len(man), man.group.nunique(), sum(inst.values()), len(set().union(*cls_split.values())), len(it), it.tray.nunique()])

# ---------- Bảng II: phân đoạn ----------
v3 = pd.read_csv(f"{K}/out_v3/summary.csv")
v4b = pd.read_csv(f"{K}/out_v4b/summary.csv")
v4f = pd.concat([pd.read_csv(f"{K}/out_v4full/summary.csv"), pd.read_csv(f"{K}/out_v4full_s02/summary.csv")])
assert len(v4f) == 3 and len(v4b) == 3
def row(d, name, val=True):
    return [name, ms(d.val_mask_mAP50_95) if val else "—", ms(d.test_mask_mAP50_95), ms(d.test_box_mAP50_95)]
def row_r(d, name):
    return [name, ms(d.test_mask_AP_excl_rice), ms(d.rice_iou), ms(d.rice_area_err * 100, 1), ms(d.mIoU_all)]
t2a = [row(v3[v3.cfg == c], n) for c, n in (("baseline", "Baseline"), ("aug", "+ Augmentation"), ("rice1", "+ Gộp cơm (rice1)"))]
t2b = [row(v4b, "Chỉ train"), row(v4f, "Train + val (cuối)", val=False)]
t2r = [row_r(v3[v3.cfg == c], n) for c, n in (("baseline", "Baseline"), ("aug", "+ Aug."), ("rice1", "+ Gộp cơm"))] + [row_r(v4b, "Chỉ train"), row_r(v4f, "Train + val")]

# ---------- Bảng III: từng món (model cuối, 3 seed) ----------
a = pd.read_csv(f"{K}/out_v4full/per_class_test_ap.csv").merge(pd.read_csv(f"{K}/out_v4full_s02/per_class_test_ap.csv"), on="lop")
ap = a.set_index("lop")[["full_s0", "full_s1", "full_s2"]]
cf = pd.read_csv(r"D:/food-labeling/grams_out/confusion_test.csv")
n_gt = cf.groupby("that").size(); n_ok = cf[cf.that == cf.doan].groupby("that").size()
n_miss = cf[cf.doan == "(bỏ sót)"].groupby("that").size()
top_wrong = cf[(cf.that != cf.doan) & (cf.doan != "(bỏ sót)")].groupby("that").doan.agg(lambda x: x.value_counts().index[0])
t3 = []
for c in ap.mean(axis=1).sort_values(ascending=False).index:
    r = ap.loc[c]
    t3.append([html.escape(VI.get(c, c)), int(n_gt.get(c, 0)), f'{r.mean():.2f}<span class="sd">±{r.std(ddof=1):.2f}</span>',
               f"{r.min():.2f}–{r.max():.2f}", f"{n_ok.get(c, 0)}/{n_gt.get(c, 0)}", int(n_miss.get(c, 0)),
               html.escape(VI.get(top_wrong.get(c, ""), top_wrong.get(c, "—"))) if c in top_wrong else "—"])
conf_ok, conf_n, conf_miss = int((cf.that == cf.doan).sum()), len(cf), int((cf.doan == "(bỏ sót)").sum())

# ---------- Bảng IV: gram (mask dự đoán, cùng 1 pipeline, cùng n / cách tính miss) ----------
F = pd.read_csv(f"{K}/out_depth_pred/depth_features_pred.csv"); F = F[(F.plane_ok == True) & (F.model == "L")]
pl = E.pl
f = it.merge(F[["image", "class", "area_px", "hsum_spread"]], on=["image", "class"], how="left")
f["hit"] = f.area_px.notna()
f["A"] = f.area_px / f.image.map(pl.plate_px); f["V"] = f.hsum_spread / f.image.map(pl.plate_px)
fs, te = f[f.split != "test"], f[f.split == "test"]

def predict(tr, va, mode):
    trh, vah = tr[tr.hit], va[va.hit]
    p = np.full(len(va), np.nan)
    if mode == "mean":
        ok = trh.groupby("class").tray.nunique(); ok = ok[ok >= E.MIN_TRAYS].index   # như grams.py: lớp < 3 khay -> TB chung
        cm = trh[trh["class"].isin(ok)].groupby("class").grams.mean()
        p[va.hit.values] = vah["class"].map(cm).fillna(trh.grams.mean()).values
        return p
    pa, _ = E.fit_predict(trh, vah, ["A"])
    pr = E.fit_predict(trh, vah, ["V"])[0] if mode == "depth" else np.full(len(vah), trh[trh["class"] == "rice"].grams.mean())
    p[va.hit.values] = np.where(vah["class"].values == "rice", pr, pa)
    return p

def score(d):
    h = d[d.hit]; e = (h.p - h.grams).abs(); r = h["class"] == "rice"
    tray = d.assign(p=d.p.fillna(0)).groupby("image").agg(p=("p", "sum"), g=("grams", "sum"))
    te_ = np.abs(tray.p - tray.g)
    return dict(n=len(d), miss=int((~d.hit).sum()), MAE=e.mean(), medAPE=np.median(e / h.grams) * 100,
                com=e[r].mean(), tru_com=e[~r].mean(), khay=te_.mean(), khay_pct=np.median(te_ / tray.g) * 100)

t4 = []
for mode, name in (("mean", "Khẩu phần TB theo món (mốc)"), ("area", "Diện tích/đĩa (cơm = TB)"), ("depth", "Diện tích/đĩa + độ sâu cho cơm")):
    cv = pd.concat([fs[fs.group == g].assign(p=predict(fs[fs.group != g], fs[fs.group == g], mode)) for g in sorted(fs.group.unique())])
    t4.append((name, score(cv), score(te.assign(p=predict(fs, te, mode)))))
for name, c, t in t4: print(name, {k: round(v, 1) for k, v in c.items()}, {k: round(v, 1) for k, v in t.items()})

# tương quan Spearman của cơm (theo khay), mask dự đoán và mask nhãn tay
r = f[(f["class"] == "rice") & f.hit].groupby("tray").mean(numeric_only=True)
sp_pred = (r.grams.rank().corr(r.A.rank()), r.grams.rank().corr(r.V.rank()), len(r))
G = pd.read_csv(f"{K}/out_depth/depth_features.csv"); G = G[G.plane_ok == True]
sp_gt = {}
for m in ("S", "L"):
    g = G[G.model == m].merge(it, on=["image", "class"]); g["A"] = g.area_px / g.image.map(pl.plate_px)
    g["V"] = g.hsum_spread / g.image.map(pl.plate_px)
    rr = g[g["class"] == "rice"].groupby("tray").mean(numeric_only=True)
    sp_gt[m] = (rr.grams.rank().corr(rr.A.rank()), rr.grams.rank().corr(rr.V.rank()), len(rr))
gs = pd.read_csv(r"D:/food-labeling/grams_out/summary.csv").set_index(["mask", "method"])
gt_upper = gs.loc[("GT-dia", "hybrid"), "test_MAE_g"]
print("spearman pred", sp_pred, "gt", sp_gt, "gt upper", gt_upper)

# ---------- Bảng VI-VII: cải tiến sau huấn luyện (research/improve) ----------
IMP = r"D:/CVision/Final_Project_Digital_Image_Processing/research/improve"
pp = pd.read_csv(f"{IMP}/result_pipeline.csv", index_col=0); ppd = pd.read_csv(f"{IMP}/result_pipeline_depth.csv", index_col=0)
rec = pd.read_csv(f"{IMP}/result_recognition.csv").set_index(["tap", "cach"])
dsm = pd.read_csv(f"{IMP}/result_downstream.csv"); mg = pd.read_csv(f"{IMP}/result_merge_test.csv")
rg = pd.read_csv(f"{IMP}/result_gram_ridge.csv").set_index("cach")
de = pd.read_csv(r"D:/food-labeling/grams_out/depth_eval.csv").set_index(["model", "cach"])
YO, DN = pp.columns
n_reg = int(pp.loc["vùng đúng"].iloc[0] + pp.loc["vùng nhầm"].iloc[0] + pp.loc["vùng sót"].iloc[0])
def pr(d, k, f="{:.0f}"): return [f.format(d.loc[k, YO]), f.format(d.loc[k, DN])]
t6 = [["Vùng nhận đúng"] + [f"{pp.loc['vùng đúng', c]:.0f} ({100 * pp.loc['vùng đúng', c] / n_reg:.1f}%)" for c in (YO, DN)],
      ["Vùng nhầm / bỏ sót"] + [f"{pp.loc['vùng nhầm', c]:.0f} / {pp.loc['vùng sót', c]:.0f}" for c in (YO, DN)],
      ["F1 danh sách món (%)"] + pr(pp, "cặp F1 %", "{:.1f}"),
      ["Món “ma” (cặp ảnh–món)"] + pr(pp, "món ma (cặp)"),
      ["Món được hỏi lại (%)"] + pr(pp, "cần xác nhận %", "{:.1f}"),
      ["Món sai không được hỏi lại"] + pr(pp, "lỗi lọt (không cờ)")]
for d in (pp, ppd):
    t6 += [["MAE gram món (g)"] + pr(d, "MAE món (g)", "{:.1f}"), ["MAE gram khay (g)"] + pr(d, "MAE gram khay", "{:.1f}"),
           ["MAE kcal khay"] + pr(d, "MAE kcal khay", "{:.1f}"), ["Trung vị sai số kcal khay (%)"] + pr(d, "medAPE kcal khay %", "{:.1f}")]
seed_rows = [r for r in rec.loc["test"].index if r.strip().startswith("full") and "YOLO" in r]
seeds = "; ".join(rec.loc[("test", r), "dung"].replace("->", " → ") for r in seed_rows)
r_te = lambda k: rec.loc[("test", k)]
# Bảng VII
def kc(model, nhan="DINO+hạ ngưỡng", gram="cũ", com="khẩu phần TB"):
    q = dsm[(dsm.com == com) & (dsm.nhan_dien == nhan) & (dsm.he_so_gram == gram)]
    return q[q.model.isin(model)].khay_kcal_MAE.mean()
three = ["full0", "full1", "full2"]
k_base, k_ens, k_ridge = kc(three), kc(["fullE"]), kc(three, gram="ridge")
sys.path.insert(0, r"D:/food-labeling/improve")
import common as CM
tta = np.mean([CM.score("val", f"tr{i}F")["dung"] for i in range(3)]), np.mean([CM.score("val", f"tr{i}T")["dung"] for i in range(3)])
val_base = float(rec.loc[("val", "YOLO (TB 3 seed)"), "dung"])
mgm = mg.groupby("cach", sort=False)[["vung_dung", "khay_kcal_MAE"]].mean()
ridge_best = rg.drop(index=[i for i in rg.index if "nhóm" in i]).cv_MAE.idxmin()
t7 = [["TTA (lật ngang, ảnh 800 px)", f"Val: vùng đúng {tta[0]:.0f} / {tta[1]:.0f} so với {val_base:.0f}", "Không cải thiện"],
      ["Gộp 3 seed", f"Test: vùng đúng {float(r_te('gộp 3 seed + DINO + hạ ngưỡng')['dung']):.0f} so với {float(r_te('+ DINO + hạ ngưỡng (TB 3 seed)')['dung']):.0f}; kcal khay {k_ens:.0f} so với {k_base:.0f}",
       "Tốn 3× YOLO, calo không tốt hơn"],
      [f"Co hệ số gram (ridge, {ridge_best.replace('ridge ', '').replace('lam', 'λ')} theo CV)",
       f"CV: MAE món khác cơm {rg.loc['hiện tại (<3 khay -> chung)', 'cv_MAE']:.1f} → {rg.loc[ridge_best, 'cv_MAE']:.1f} g; test: kcal khay {k_base:.0f} → {k_ridge:.0f}", "Calo tổng khay tệ hơn"],
      ["Gộp 4 cặp món gần giống", f"Vùng đúng (mức nhóm) {mgm.loc['không gộp, chấm theo nhóm A', 'vung_dung']:.0f} → {mgm.loc['A (4 cặp, kcal gần)', 'vung_dung']:.0f}; kcal khay {mgm.loc['không gộp', 'khay_kcal_MAE']:.0f} → {mgm.loc['A (4 cặp, kcal gần)', 'khay_kcal_MAE']:.0f}",
       "Chỉ đổi cách chấm, không nhận tốt hơn"],
      ["Độ sâu cho món khác cơm", f"CV: MAE món khác cơm {de.loc[('L', 'area'), 'cv_MAE_tru_com']:.1f} → {de.loc[('L', 'both_spread'), 'cv_MAE_tru_com']:.1f} g", "Không giúp"]]
# độ phủ của "±30%" (phần dư CV và test, món nhận đúng; như research/improve/interval.py)
import depth_eval_pred as DP
cov = {}
fi = DP.it.merge(DP.F[["image", "class", "area_px", "hsum_spread"]], on=["image", "class"], how="left")
fi["hit"] = fi.area_px.notna(); fi["A"] = fi.area_px / fi.image.map(pl.plate_px); fi["V"] = fi.hsum_spread / fi.image.map(pl.plate_px)
fsi, tei = fi[fi.split != "test"], fi[fi.split == "test"]
for rv in (False, True):
    cvi = pd.concat([fsi[fsi.group == g].assign(p=DP.hyb(fsi[fsi.group != g], fsi[fsi.group == g], rv)) for g in sorted(fsi.group.unique())])
    for nm, dd in (("cv", cvi), ("te", tei.assign(p=DP.hyb(fsi, tei, rv)))):
        dd = dd[dd.hit & dd.p.notna()]; x = (dd.grams / dd.p - 1).abs(); r = dd["class"] == "rice"
        if not rv:
            cov[f"nr30_{nm}"], cov[f"nr50_{nm}"] = 100 * (x[~r] <= .3).mean(), 100 * (x[~r] <= .5).mean()
        else:
            cov[f"rd30_{nm}"] = 100 * (x[r] <= .3).mean()
print("cov", {k: round(v) for k, v in cov.items()}, "tta", tta)

# ---------- Hình 1 ----------
im = cv2.imread(f"{K}/out_depth/vis/B_005_h.jpg"); im = cv2.resize(im, (1200, int(im.shape[0] * 1200 / im.shape[1])))
cv2.imwrite(f"{OUT}/fig_depth.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])

# ---------- HTML ----------
def tbl(head, rows, bold_last=False, groups=None):
    h = "<tr>" + "".join(f"<th>{x}</th>" for x in head) + "</tr>"
    body = ""
    for i, rw in enumerate(rows):
        if groups and i in groups:
            body += f'<tr class="grp"><td colspan="{len(head)}">{groups[i]}</td></tr>'
        cls = ' class="b"' if bold_last and i == len(rows) - 1 else ""
        body += f"<tr{cls}>" + "".join(f"<td>{x}</td>" for x in rw) + "</tr>"
    return f"<table><thead>{h}</thead><tbody>{body}</tbody></table>"

T1 = tbl(["Tập", "Ảnh", "Nhóm buổi chụp", "Đa giác", "Số lớp có mặt", "Cặp có cân", "Khay có cân"], t1, bold_last=True)
G2 = {0: "Bộ nhãn v3 (47 lớp)", 3: "Bộ nhãn v4 (45 lớp), aug + rice1"}
T2 = tbl(["Cấu hình", "Val mask", "Test mask", "Test box"], t2a + t2b, groups=G2)
T2R = tbl(["Cấu hình", "Mask trừ cơm", "IoU cơm", "Sai số DT cơm (%)", "mIoU"], t2r, groups=G2)
T3 = tbl(["Món", "GT", "AP mask", "Min–max", "Đúng", "Sót", "Hay nhầm thành"], t3)
def g4(i, k, d=1): return f"{t4[i][1][k]:.{d}f}", f"{t4[i][2][k]:.{d}f}"
rows4 = []
for k, name in (("MAE", "MAE món (g)"), ("medAPE", "Trung vị sai số món (%)"), ("tru_com", "MAE món khác cơm (g)"),
                ("com", "MAE cơm (g)"), ("khay", "MAE tổng khay (g)"), ("khay_pct", "Trung vị sai số khay (%)")):
    rows4.append([name] + [x for i in range(3) for x in g4(i, k)])
T6 = tbl(["Chỉ số", "Chỉ YOLO", "+ DINOv2"], t6, groups={0: f"Nhận diện (36 ảnh, {n_reg} vùng nhãn)", 6: "Khối lượng và calo, cơm = khẩu phần TB",
                                                       10: "Khối lượng và calo, cơm = độ sâu"})
T7 = tbl(["Hướng thử", "Kết quả", "Lý do bỏ"], t7).replace("<table>", '<table class="wrap">')
T4 = ("<table><thead><tr><th rowspan=2>Chỉ số</th><th colspan=2>Mốc TB</th><th colspan=2>Diện tích</th><th colspan=2>DT + độ sâu</th></tr>"
      "<tr><th>CV</th><th>Test</th><th>CV</th><th>Test</th><th>CV</th><th>Test</th></tr></thead><tbody>"
      + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in rw) + "</tr>" for rw in rows4) + "</tbody></table>")

n_te, miss_te, n_cv, miss_cv = t4[0][2]["n"], t4[0][2]["miss"], t4[0][1]["n"], t4[0][1]["miss"]
v3m = {c: v3[v3.cfg == c] for c in ("baseline", "aug", "rice1")}
D = dict(
    n_img=len(man), n_cls=45, n_test_cls=len(cls_split["test"]), n_pairs=len(it), n_trays=it.tray.nunique(),
    te_trays=trays["test"], fit_trays=it[it.split != "test"].tray.nunique(),
    base=ms(v3m["baseline"].test_mask_mAP50_95), rice1=ms(v3m["rice1"].test_mask_mAP50_95), aug=ms(v3m["aug"].test_mask_mAP50_95),
    full=ms(v4f.test_mask_mAP50_95), full_box=f"{v4f.test_box_mAP50_95.mean():.3f}", full_rice=f"{(v4f.rice_area_err * 100).mean():.1f}",
    trainonly=ms(v4b.test_mask_mAP50_95),
    conf_ok=conf_ok, conf_n=conf_n, conf_pct=f"{conf_ok / conf_n * 100:.0f}", conf_miss=conf_miss,
    m0=t4[0][2]["MAE"], m1=t4[1][2]["MAE"], m2=t4[2][2]["MAE"], nr0=t4[0][2]["tru_com"], nr1=t4[1][2]["tru_com"],
    k1=t4[1][2]["khay"], k2=t4[2][2]["khay"], kc1=t4[1][1]["khay"], kc2=t4[2][1]["khay"], c1=t4[1][1]["com"], c2=t4[2][1]["com"],
    kp1=t4[1][1]["khay_pct"], kp2=t4[2][1]["khay_pct"],
    n_te=n_te, miss_te=miss_te, n_cv=n_cv, miss_cv=miss_cv, gt_upper=gt_upper,
    spA=sp_pred[0], spV=sp_pred[1], spn=sp_pred[2], gA=sp_gt["L"][0], gS=sp_gt["S"][1], gL=sp_gt["L"][1], gn=sp_gt["L"][2],
    fb=ap.loc["fish_ball"], T1=T1, T2=T2, T2R=T2R, T3=T3, T4=T4, T6=T6, T7=T7,
    d_ok0=pp.loc["vùng đúng", YO], d_ok1=pp.loc["vùng đúng", DN], d_n=n_reg, d_f0=pp.loc["cặp F1 %", YO], d_f1=pp.loc["cặp F1 %", DN],
    d_k0=pp.loc["MAE kcal khay", YO], d_k1=pp.loc["MAE kcal khay", DN], d_kd0=ppd.loc["MAE kcal khay", YO], d_kd1=ppd.loc["MAE kcal khay", DN],
    d_g0=pp.loc["MAE gram khay", YO], d_g1=pp.loc["MAE gram khay", DN], d_l0=pp.loc["lỗi lọt (không cờ)", YO], d_l1=pp.loc["lỗi lọt (không cờ)", DN],
    d_s0=pp.loc["vùng sót", YO], d_s1=pp.loc["vùng sót", DN], d_q0=pp.loc["cần xác nhận %", YO], d_q1=pp.loc["cần xác nhận %", DN],
    d_gd0=ppd.loc["MAE gram khay", YO], d_gd1=ppd.loc["MAE gram khay", DN],
    v_y=float(rec.loc[("val", "YOLO (TB 3 seed)"), "dung"]), v_d=float(rec.loc[("val", "+ DINO + hạ ngưỡng (TB 3 seed)"), "dung"]),
    t_y=float(r_te("YOLO (TB 3 seed)")["dung"]), t_d=float(r_te("+ DINO + hạ ngưỡng (TB 3 seed)")["dung"]), seeds=seeds, **cov)

import re
tpl = open(f"{OUT}/template.html", encoding="utf-8").read()
def sub(m):
    expr, _, fmt = m.group(1).partition("|")
    v = eval(expr, {}, D)
    out = format(v, fmt) if fmt else str(v)
    return out.replace("-", "−") if isinstance(v, (int, float, np.floating)) else out
open(f"{OUT}/benchmark.html", "w", encoding="utf-8").write(re.sub(r"\[\[(.+?)\]\]", sub, tpl))
print("html ok")
