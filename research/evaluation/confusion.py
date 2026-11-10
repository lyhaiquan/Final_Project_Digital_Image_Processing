"""Món nào bị nhầm thành món gì (tập test, model cuối).
Mỗi polygon nhãn tay -> lớp dự đoán chiếm nhiều diện tích nhất bên trong nó (>=30% diện tích polygon), không có -> "bỏ sót".
    python confusion.py -> grams_out/confusion_test.csv + grams_out/confusion_test.jpg
"""
import sys, glob, collections, numpy as np, pandas as pd, cv2
from ultralytics import YOLO
sys.stdout.reconfigure(encoding="utf-8")
D, OUT = r"D:/food_dataset_v4", r"D:/food-labeling/grams_out"
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
model = YOLO(r"D:/food-labeling/models/food_seg_v4_final.pt")
rows, crops = [], collections.defaultdict(list)
for p in sorted(glob.glob(f"{D}/images/test/*.jpg")):
    im = p.split("\\")[-1].split("/")[-1]
    img = cv2.imread(p); H, W = img.shape[:2]
    r = model.predict(p, conf=0.25, retina_masks=True, verbose=False)[0]
    pm = [(model.names[int(c)], m > 0.5) for m, c in zip(r.masks.data.cpu().numpy(), r.boxes.cls)] if r.masks is not None else []
    for l in open(f"{D}/labels/test/{im[:-4]}.txt"):
        t = l.split()
        if not t: continue
        g = NAMES[int(t[0])]; xy = (np.array(t[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
        gm = np.zeros((H, W), np.uint8); cv2.fillPoly(gm, [xy], 1); gm = gm > 0
        cov = collections.Counter()
        for c, m in pm: cov[c] = max(cov[c], (m & gm).sum() / gm.sum())
        best, f = (cov.most_common(1)[0] if cov else ("", 0))
        pred = best if f >= 0.3 else "(bỏ sót)"
        rows.append(dict(image=im, that=g, doan=pred, phu=round(f, 2)))
        if pred != g and len(crops[g]) < 4:
            x, y, w, h = cv2.boundingRect(xy); pad = int(0.15 * max(w, h))
            cr = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].copy()
            cv2.polylines(cr, [xy - [max(0, x - pad), max(0, y - pad)]], True, (0, 255, 255), max(2, W // 300))
            s = 220 / max(cr.shape[:2]); cr = cv2.resize(cr, (int(cr.shape[1] * s), int(cr.shape[0] * s)))
            cr = cv2.copyMakeBorder(cr, 0, 220 - cr.shape[0], 0, 220 - cr.shape[1], cv2.BORDER_CONSTANT)
            cv2.rectangle(cr, (0, 0), (220, 34), (0, 0, 0), -1)
            cv2.putText(cr, f"that: {g[:22]}", (3, 13), 0, 0.42, (0, 255, 0), 1)
            cv2.putText(cr, f"doan: {pred[:22]}", (3, 29), 0, 0.42, (0, 128, 255), 1)
            crops[g].append(cr)
d = pd.DataFrame(rows); d.to_csv(f"{OUT}/confusion_test.csv", index=False)
s = d.groupby("that").agg(n=("doan", "size"), dung=("doan", lambda x: 0)).reset_index()
s["dung"] = [((d.that == g) & (d.doan == g)).sum() for g in s.that]
s["nham_thanh"] = [", ".join(f"{k} ({v})" for k, v in d[(d.that == g) & (d.doan != g)].doan.value_counts().items()) for g in s.that]
s["sai_%"] = (100 * (1 - s.dung / s.n)).round(0)
s = s.sort_values(["sai_%", "n"], ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(s.to_string(index=False)); print("tổng polygon:", len(d), "| đúng:", (d.that == d.doan).sum())
order = [g for g in s.that if crops[g]]
rows_img = [np.hstack(crops[g] + [np.zeros_like(crops[g][0])] * (4 - len(crops[g]))) for g in order]
cv2.imwrite(f"{OUT}/confusion_test.jpg", np.vstack(rows_img))
