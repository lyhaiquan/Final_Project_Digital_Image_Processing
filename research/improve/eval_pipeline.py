"""Chấm pipeline THẬT trong repo (calo_khay.CaloKhay) trên tập test khóa: cũ (chỉ YOLO) so với mới (thêm DINOv2).

Nhận diện: mỗi polygon nhãn tay -> lớp dự đoán phủ nhiều nhất (>= 30% diện tích polygon), không có -> bỏ sót.
Gram / kcal: tổng khay tính trên MỌI món dự đoán (kể cả món "ma"); món có nhãn nhưng không được cân thì bỏ qua.
Cần bộ dữ liệu gốc: D:/food_dataset_v4.

    python research/improve/eval_pipeline.py [--depth]
"""
import os, sys
import cv2, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from calo_khay import CaloKhay  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = "D:/food_dataset_v4"
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
man = pd.read_csv(f"{D}/manifest.csv"); test = sorted(man[man.split == "test"].image)
w = pd.read_csv(f"{D}/weights.csv"); w = w[w.image.isin(test)].groupby(["image", "class"]).grams.sum().reset_index()


def run(ck):
    cal = ck.kcal100
    ok = wrong = miss = tp = fp = fn = flagged = n_items = caught = errs = 0
    rows = []
    for im in test:
        bgr = cv2.imread(f"{D}/images/test/{im}"); H, W = bgr.shape[:2]
        res = ck.analyze(bgr)
        pm = {}
        for it in res["items"]:
            m = np.zeros((H, W), np.uint8)
            for p in it["polygons"]: cv2.fillPoly(m, [np.array(p, np.int32)], 1)
            pm[it["cls"]] = m > 0
        labs = set()
        for l in open(f"{D}/labels/test/{im[:-4]}.txt"):
            t = l.split()
            if not t: continue
            g = NAMES[int(t[0])]; labs.add(g)
            gm = np.zeros((H, W), np.uint8); cv2.fillPoly(gm, [(np.array(t[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)], 1); gm = gm > 0
            cov = {c: (m & gm).sum() / max(gm.sum(), 1) for c, m in pm.items()}
            best = max(cov, key=cov.get) if cov else None
            pred = best if best and cov[best] >= 0.3 else None
            ok += pred == g; miss += pred is None; wrong += pred not in (g, None)
        ps = set(pm); tp += len(labs & ps); fp += len(ps - labs); fn += len(labs - ps)
        for it in res["items"]:
            e = it["cls"] not in labs
            n_items += 1; flagged += it["needs_confirm"]; errs += e; caught += e and it["needs_confirm"]
        wt = w[w.image == im]
        if len(wt):
            true = dict(zip(wt["class"], wt.grams))
            pred = {it["cls"]: it for it in res["items"] if not (it["cls"] in labs and it["cls"] not in true)}
            hit = [abs(pred[c]["grams"] - g) for c, g in true.items() if c in pred]
            rows.append(dict(image=im, n=len(true), sot=sum(c not in pred for c in true), ma=sum(c not in true for c in pred), e_mon=sum(hit), n_hit=len(hit),
                             g=sum(true.values()), p=sum(i["grams"] for i in pred.values()),
                             k=sum(g * cal.get(c, 0) / 100 for c, g in true.items()), pk=sum(i["kcal"] for i in pred.values())))
    r = pd.DataFrame(rows); P, R = tp / (tp + fp), tp / (tp + fn)
    return {"vùng đúng": ok, "vùng nhầm": wrong, "vùng sót": miss, "cặp F1 %": 200 * P * R / (P + R), "món ma (cặp)": fp,
            "cần xác nhận %": 100 * flagged / n_items, "lỗi được gắn cờ %": 100 * caught / max(errs, 1), "lỗi lọt (không cờ)": errs - caught,
            "cặp cân bị sót": r.sot.sum(), "cặp ma (ảnh có cân)": r.ma.sum(), "MAE món (g)": r.e_mon.sum() / r.n_hit.sum(),
            "MAE gram khay": (r.p - r.g).abs().mean(), "lệch gram khay": (r.p - r.g).mean(),
            "MAE kcal khay": (r.pk - r.k).abs().mean(), "medAPE kcal khay %": ((r.pk - r.k).abs() / r.k).median() * 100, "lệch kcal khay": (r.pk - r.k).mean()}


if __name__ == "__main__":
    depth = "--depth" in sys.argv
    out = {}
    for name, dino in (("cũ (chỉ YOLO)", False), ("mới (+DINOv2)", True)):
        ck = CaloKhay(use_depth=depth, use_dino=dino); out[name] = run(ck); del ck
    d = pd.DataFrame(out).round(1); pd.set_option("display.width", 200)
    print(f"TEST: {len(test)} ảnh, 382 polygon; cơm = {'độ sâu' if depth else 'khẩu phần TB'}"); print(d.to_string())
    d.to_csv(os.path.join(HERE, f"result_pipeline{'_depth' if depth else ''}.csv"))
