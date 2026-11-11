"""Cache dự đoán (mask, lớp, conf>=0.05) của 6 model, 3 kiểu ảnh vào: n640 (gốc), f640 (lật ngang), n800 (ảnh 800).
    python cache_pred.py
"""
import os, numpy as np, cv2, torch
from ultralytics import YOLO
from common import *

JOBS = [(t, "val") for t in ("tr0", "tr1", "tr2")] + [(t, "test") for t in ("full0", "full1", "full2")]
AUGS = {"n640": (640, False), "f640": (640, True), "n800": (800, False)}
for tag, split in JOBS:
    model = YOLO(MODELS[tag])
    for aug, (sz, flip) in AUGS.items():
        if os.path.isdir(f"{CACHE}/pred/{tag}_{aug}") and len(os.listdir(f"{CACHE}/pred/{tag}_{aug}")) == len(images(split)): continue
        for im in images(split):
            a = read(split, im); H, W = a.shape[:2]
            r = model.predict(cv2.flip(a, 1) if flip else a, conf=0.05, imgsz=sz, retina_masks=True, verbose=False)[0]
            if r.masks is None:
                m = np.zeros((0, H, W), bool); c = np.zeros(0, int); s = np.zeros(0)
            else:
                m = r.masks.data.cpu().numpy() > 0.5
                if flip: m = m[:, :, ::-1]
                c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy()
            save_pred(tag, split, im, aug, np.ascontiguousarray(m), c, s)
        print(tag, aug, "xong", flush=True)
    del model; torch.cuda.empty_cache()

# kiểm tra: baseline phải khớp bảng cũ (full1 trên test: 266/382 đúng, 37 sót)
for tag, split in JOBS:
    print(tag, split, score(split, lambda im: load_pred(tag, split, im)))
