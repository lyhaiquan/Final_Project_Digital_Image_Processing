"""Diện tích đĩa (pixel) cho mỗi ảnh, dùng làm thước đo tỉ lệ.

YOLOE (open-vocab) với prompt "plate" -> lấy mask đĩa có conf cao nhất, hợp với mask đồ ăn nhãn tay
(đồ ăn nằm TRÊN đĩa nên mask đĩa có thể bị khuyết) -> lấy viền ngoài -> bỏ các điểm sát mép ảnh
(đĩa bị khung cắt) -> fit elip -> diện tích elip = diện tích đĩa kể cả phần ngoài khung.

    python plate.py   -> grams_out/plate_areas.csv + grams_out/plate_check.jpg
"""
import os, sys, numpy as np, pandas as pd, cv2
sys.stdout.reconfigure(encoding="utf-8")
D, OUT = r"D:/food_dataset_v4", r"D:/food-labeling/grams_out"
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{D}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]


def food_mask(lbl, H, W):
    m = np.zeros((H, W), np.uint8)
    for l in open(lbl):
        r = l.split()
        if r: cv2.fillPoly(m, [(np.array(r[1:], float).reshape(-1, 2) * [W, H]).round().astype(np.int32)], 1)
    return m


def plate_area(plate, food, H, W, margin=4):
    """Bao lồi của (mask đĩa ∪ đồ ăn). Nếu đĩa bị khung ảnh cắt (>2% viền nằm sát mép) thì fit elip
    lên phần viền KHÔNG sát mép để bù phần bị cắt; chỉ nhận elip nếu nó lớn hơn bao lồi và < 2 lần."""
    m = ((plate > 0) | (food > 0)).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    hm = np.zeros((H, W), np.uint8); cv2.fillPoly(hm, [cv2.convexHull(c)], 1)
    hull_a = float(hm.sum())
    dense = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0].reshape(-1, 2)
    keep = (dense[:, 0] > margin) & (dense[:, 0] < W - 1 - margin) & (dense[:, 1] > margin) & (dense[:, 1] < H - 1 - margin)
    cut = 1 - keep.mean()
    if cut > 0.02 and keep.sum() >= 50:
        el = cv2.fitEllipse(dense[keep].astype(np.float32))
        ea = np.pi * el[1][0] * el[1][1] / 4
        if hull_a <= ea <= 2 * hull_a: return ea, hull_a, cut, el
    return hull_a, hull_a, cut, None


def main():
    from ultralytics import YOLOE, YOLO
    seg = YOLO(r"D:/food-labeling/models/food_seg_v4_final.pt")
    os.chdir(r"D:/food-labeling/models/yoloe")
    model = YOLOE("yoloe-11s-seg.pt"); model.set_classes(["plate"])
    man = pd.read_csv(f"{D}/manifest.csv")
    imgs = set(pd.read_csv(f"{D}/weights.csv").image)
    rows, tiles = [], []
    for im, sp in man[man.image.isin(imgs)][["image", "split"]].itertuples(index=False):
        p = f"{D}/images/{sp}/{im}"
        r = model.predict(p, conf=0.1, retina_masks=True, verbose=False)[0]
        H, W = r.orig_shape
        plates = list(r.masks.data.cpu().numpy() > 0.5) if r.masks is not None else []
        hulls = []   # mask "plate" của YOLOE chỉ là phần mặt đĩa LỘ RA (đồ ăn bị trừ) -> lấp bằng bao lồi
        for pl in plates:
            hm = np.zeros((H, W), np.uint8)
            cs = cv2.findContours(pl.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if cs: cv2.fillPoly(hm, [cv2.convexHull(np.vstack(cs))], 1)
            hulls.append(hm > 0)
        foods = {"gt": food_mask(f"{D}/labels/{sp}/{im[:-4]}.txt", H, W)}
        q = seg.predict(p, conf=0.25, retina_masks=True, verbose=False)[0]
        foods["pred"] = (q.masks.data.cpu().numpy() > 0.5).any(0).astype(np.uint8) if q.masks is not None else np.zeros((H, W), np.uint8)
        row = dict(image=im, H=H, W=W, n_plates=len(plates))
        # chọn đĩa = mask "plate" (đã lấp bằng bao lồi) chứa nhiều đồ ăn DỰ ĐOÁN nhất -> không dùng nhãn tay
        food = foods["pred"] if foods["pred"].any() else foods["gt"]
        ov = [(hl & (food > 0)).sum() / max(food.sum(), 1) for hl in hulls]
        k = int(np.argmax(ov)) if plates else -1
        if k >= 0 and ov[k] > 0.3:
            # diện tích CHỈ từ đĩa (không hợp với đồ ăn: mask dự đoán lan ra ngoài làm bao lồi phình)
            ea, ha, cut, el = plate_area(hulls[k], np.zeros((H, W), np.uint8), H, W)
            plate = plates[k]
        else:
            ea = ha = cut = np.nan; el = None; plate = np.zeros((H, W), bool)
        row.update(plate_px=ea, hull_px=ha, cut_frac=cut, food_on_plate=ov[k] if k >= 0 else 0,
                   food_on_plate_gt=((hulls[k] & (foods["gt"] > 0)).sum() / foods["gt"].sum()) if k >= 0 else 0)
        a = cv2.imread(p); a[plate] = (0.5 * a[plate] + [127, 0, 0]).astype(np.uint8)
        if el is not None: cv2.ellipse(a, el, (0, 255, 255), max(3, W // 200))
        # mặt đĩa LỘ RA (mask YOLOE đã chọn) -> mặt phẳng nền cho bước độ sâu (depth_check)
        os.makedirs(f"{OUT}/plate_masks", exist_ok=True)
        cv2.imwrite(f"{OUT}/plate_masks/{im[:-4]}.png", plate.astype(np.uint8) * 255)
        rows.append(row)
        s_ = 300 / max(H, W); a = cv2.resize(a, (int(W * s_), int(H * s_)))
        a = cv2.copyMakeBorder(a, 0, 300 - a.shape[0], 0, 300 - a.shape[1], cv2.BORDER_CONSTANT)
        cv2.putText(a, f"{im} {row['plate_px'] / (H * W):.2f}", (4, 16), 0, 0.45, (0, 255, 0), 1); tiles.append(a)
    d = pd.DataFrame(rows); d.to_csv(f"{OUT}/plate_areas.csv", index=False)
    while len(tiles) % 10: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(f"{OUT}/plate_check.jpg", np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))
    print(d.describe().round(2).to_string()); print(d.isna().sum().to_dict())


if __name__ == "__main__":
    main()
