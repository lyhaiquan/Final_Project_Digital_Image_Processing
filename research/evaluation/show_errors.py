"""Ảnh test có món bị nhầm: trái = nhãn tay (viền vàng = món bị sai, ghi "thật -> đoán"), phải = model dự đoán.
Bỏ qua mảnh cơm nhỏ (<1% ảnh).  -> grams_out/errors/sheet_XX.jpg"""
import os, sys, glob, collections, numpy as np, pandas as pd, cv2
from ultralytics import YOLO
sys.stdout.reconfigure(encoding="utf-8")
D, OUT = r"D:/food_dataset_v4", r"D:/food-labeling/grams_out/errors"
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(f"{OUT}/*.jpg"): os.remove(f)
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
model = YOLO(r"D:/food-labeling/models/food_seg_v4_final.pt")
S = 560; panels = []
for p in sorted(glob.glob(f"{D}/images/test/*.jpg")):
    im = os.path.basename(p); img = cv2.imread(p); H, W = img.shape[:2]
    r = model.predict(p, conf=0.25, retina_masks=True, verbose=False)[0]
    pm = [(model.names[int(c)], m > 0.5) for m, c in zip(r.masks.data.cpu().numpy(), r.boxes.cls)] if r.masks is not None else []
    left = img.copy(); errs = []
    for l in open(f"{D}/labels/test/{im[:-4]}.txt"):
        t = l.split()
        if not t: continue
        g = NAMES[int(t[0])]; xy = (np.array(t[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)
        gm = np.zeros((H, W), np.uint8); cv2.fillPoly(gm, [xy], 1); gm = gm > 0
        cov = collections.Counter()
        for c, m in pm: cov[c] = max(cov[c], (m & gm).sum() / gm.sum())
        best, f = cov.most_common(1)[0] if cov else ("", 0)
        pred = best if f >= 0.3 else "bo sot"
        if pred == g or (g == "rice" and gm.sum() < 0.01 * H * W): continue
        errs.append((g, pred, xy))
    if not errs: continue
    th = max(3, W // 250); fs = W / 900
    for i, (g, pred, xy) in enumerate(errs, 1):
        cv2.polylines(left, [xy], True, (0, 255, 255), th)
        x, y = xy.mean(0).astype(int)
        cv2.putText(left, str(i), (x, y), 0, 1.6 * fs, (0, 0, 0), int(8 * fs) + 2); cv2.putText(left, str(i), (x, y), 0, 1.6 * fs, (0, 255, 255), int(3 * fs) + 1)
    right = r.plot(line_width=th, font_size=max(12, W // 60))
    s = S / max(H, W)
    pan = np.hstack([cv2.resize(left, (int(W * s), int(H * s))), cv2.resize(right, (int(W * s), int(H * s)))])
    pan = cv2.copyMakeBorder(pan, 0, S - pan.shape[0], 0, 2 * S - pan.shape[1], cv2.BORDER_CONSTANT)
    leg = np.zeros((26 + 20 * len(errs), 2 * S, 3), np.uint8)
    cv2.putText(leg, f"{im}   TRAI: nhan tay (vien vang = mon bi sai)   PHAI: model du doan", (6, 18), 0, 0.5, (255, 255, 255), 1)
    for i, (g, pred, _) in enumerate(errs, 1):
        cv2.putText(leg, f"{i}. that: {g}  ->  doan: {pred}", (6, 18 + 20 * i), 0, 0.5, (0, 255, 255), 1)
    panels.append(np.vstack([leg, pan, np.full((8, 2 * S, 3), 80, np.uint8)]))
    print(im, [(g, pred) for g, pred, _ in errs])
k = 0
for i in range(0, len(panels), 4):
    k += 1; cv2.imwrite(f"{OUT}/sheet_{k:02d}.jpg", np.vstack(panels[i:i + 4]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(panels), "ảnh ->", k, "tờ")
