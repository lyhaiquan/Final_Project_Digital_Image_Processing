"""Hướng 3: co hệ số gram của từng món về trung bình (ridge) thay cho quy tắc cứng "< 3 khay -> hệ số chung".
Món khác cơm: log g = mu + a_món + b·log A, phạt lam·a_món² (lam=0: mỗi món tự do; lam lớn: mọi món dùng chung).
Biến thể nhóm: a_món co về trung bình NHÓM món (rau / thịt / trứng / chiên...).
CV để riêng 1 buổi chụp trên train+val (mask dự đoán của model cuối), test chấm một lần với lam chọn theo CV.
"""
import sys, numpy as np, pandas as pd
from common import *
sys.path.insert(0, ROOT)
import depth_eval as DE, depth_eval_pred as DP

GROUP = dict(rau="bean_sprouts_bamboo pumpkin leafy_greens chayote radish water_spinach stir_fried_vegetables bean_sprouts_green_beans cabbage potato soup",
             thit="betel_leaf_pork minced_pork_corn bell_pepper_chicken coconut_braised_pork fried_chicken pork_broccoli cutlet boiled_pork sweet_sour_chicken dog_meat_rua_man pork_with_tofu stir_fried_gizzard roasted_cinnamon_pork braised_fish silkworm_pupae",
             trung="rolled_omelette boiled_egg quail_egg egg_meatloaf braised_egg fried_egg",
             che_bien="chinese_sausage sausage pork_roll shrimp_spring_roll fish_ball fried_tofu fried_shrimp fried_pork_skin",
             kho="pork_floss fried_anchovy peanuts drink")
G_OF = {c: g for g, cs in GROUP.items() for c in cs.split()}


def fit_predict(tr, te, lam, grouped):
    cls = sorted(set(tr["class"])); grp = sorted(set(G_OF.values()))
    def X(d):
        x = np.zeros((len(d), 2 + len(cls) + (len(grp) if grouped else 0)))
        x[:, 0] = 1; x[:, 1] = np.log(np.clip(d.A.values, 1e-6, None))
        for i, c in enumerate(d["class"]):
            if c in cls: x[i, 2 + cls.index(c)] = 1
            if grouped: x[i, 2 + len(cls) + grp.index(G_OF[c])] = 1
        return x
    Xt, y = X(tr), np.log(tr.grams.values)
    pen = np.zeros(Xt.shape[1]); pen[2:2 + len(cls)] = lam
    if grouped: pen[2 + len(cls):] = 1e-3      # nhóm gần như tự do
    coef = np.linalg.solve(Xt.T @ Xt + np.diag(pen) + 1e-9 * np.eye(Xt.shape[1]), Xt.T @ y)
    s = np.mean(np.exp(y - Xt @ coef))
    return np.exp(X(te) @ coef) * s


def main():
    f = DP.it.merge(DP.F[["image", "class", "area_px"]], on=["image", "class"], how="left")
    f["hit"] = f.area_px.notna(); f["A"] = f.area_px / f.image.map(DP.pl.plate_px)
    f = f[f.hit & (f["class"] != "rice") & f.A.notna()]
    fs, te = f[f.split != "test"], f[f.split == "test"]
    groups = sorted(fs.group.unique())
    rows = []
    cur = pd.concat([fs[fs.group == g].assign(p=DE.fit_predict(fs[fs.group != g], fs[fs.group == g], ["A"])[0]) for g in groups])
    rows.append(dict(cach="hiện tại (<3 khay -> chung)", cv_MAE=(cur.p - cur.grams).abs().mean(), cv_medAPE=((cur.p - cur.grams).abs() / cur.grams).median() * 100,
                     test_MAE=np.abs(DE.fit_predict(fs, te, ["A"])[0] - te.grams).mean()))
    for grouped in (False, True):
        for lam in (0, 0.3, 1, 3, 10, 30, 1e6):
            cv = pd.concat([fs[fs.group == g].assign(p=fit_predict(fs[fs.group != g], fs[fs.group == g], lam, grouped)) for g in groups])
            rows.append(dict(cach=f"ridge lam={lam:g}" + (" + nhóm món" if grouped else ""), cv_MAE=(cv.p - cv.grams).abs().mean(),
                             cv_medAPE=((cv.p - cv.grams).abs() / cv.grams).median() * 100, test_MAE=np.abs(fit_predict(fs, te, lam, grouped) - te.grams).mean()))
    d = pd.DataFrame(rows); pd.set_option("display.width", 200); print(f"món khác cơm: {len(fs)} cặp CV, {len(te)} cặp test"); print(d.round(2).to_string(index=False))
    d.to_csv(f"{ROOT}/improve/result_gram_ridge.csv", index=False)


if __name__ == "__main__":
    main()
