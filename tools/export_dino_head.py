"""Fit đầu phân loại DINOv2 MỘT lần trên polygon nhãn tay train+val rồi lưu ra models/dino_head.npz.

Dùng chính calo_khay.pipeline.RegionClassifier để trích đặc trưng, nên lúc fit và lúc chạy giống hệt nhau.
Cần bộ dữ liệu gốc (không có trong repo): D:/food_dataset_v4. Cần thêm scikit-learn.
Cấu hình (DINOv2-base 224 px, đặc trưng trong mask, logistic C=100) được chọn trên tập val: xem research/improve/.

    python tools/export_dino_head.py
"""
import os, sys
import cv2, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from calo_khay.pipeline import RegionClassifier  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
D = "D:/food_dataset_v4"
OUT = os.path.join(HERE, "..", "models", "dino_head.npz")
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
man = pd.read_csv(f"{D}/manifest.csv")

rc = RegionClassifier(load_head=False)
X, y = [], []
for im, sp in man[man.split.isin(["train", "val"])][["image", "split"]].itertuples(index=False):
    bgr = cv2.imread(f"{D}/images/{sp}/{im}"); H, W = bgr.shape[:2]
    masks = []
    for l in open(f"{D}/labels/{sp}/{im[:-4]}.txt"):
        t = l.split()
        if not t: continue
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [(np.array(t[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)], 1)
        if m.any(): masks.append(m > 0); y.append(int(t[0]))
    X.append(rc.features(bgr, masks))
X, y = np.concatenate(X), np.array(y)
clf = LogisticRegression(C=100, max_iter=5000).fit(X, y)
# kiểm tra: softmax tự viết (như trong pipeline) phải khớp predict_proba của sklearn
z = X[:50] @ clf.coef_.T + clf.intercept_; z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
assert np.allclose(z, clf.predict_proba(X[:50]), atol=1e-5)
np.savez(OUT, coef=clf.coef_.astype(np.float32), intercept=clf.intercept_.astype(np.float32),
         names=np.array([NAMES[c] for c in clf.classes_]))
print(f"{len(y)} polygon, {len(clf.classes_)} lớp có mẫu / {len(NAMES)} -> {os.path.normpath(OUT)} ({os.path.getsize(OUT) / 1024:.0f} KB)")
print("lớp không có mẫu (DINO sẽ không bao giờ chọn):", [n for i, n in enumerate(NAMES) if i not in set(clf.classes_)])
