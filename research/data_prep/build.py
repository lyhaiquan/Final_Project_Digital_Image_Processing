"""Mask SAM (cache/) -> polygon không chồng lấn -> ảnh xem trước + các định dạng nhãn.

Dùng:
    python build.py preview 0 1 2     # chỉ vẽ ảnh xem trước để soát (nhanh)
    python build.py export            # xuất toàn bộ vào D:/Image_dataset_labeled
"""
import csv
import json
import os
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import CACHE, FILES, ROOT, SRC, parse_prompts, prompt_hash  # noqa: E402
from weights_config import MEALS, READINGS, weights_for  # noqa: E402

OUT = r"D:\Image_dataset_labeled"
PREVIEW_DIR = os.path.join(ROOT, "preview")
MIN_AREA_FRAC = 0.0003      # bỏ mảnh vụn < 0.03% diện tích ảnh (~370 px ở 1280x960)
HOLE_FRAC = 0.002           # lấp lỗ thủng nhỏ < 0.2% diện tích ảnh bên trong một món
SIMPLIFY_PX = 1.2           # dung sai rút gọn polygon (pixel)

# Món có cân nặng suy ra bằng phép trừ thay vì cân trực tiếp
DERIVED = {("B", "rice"): "trừ đĩa đen 167 g (đĩa suy từ bữa C)",
           ("C", "rice"): "587 − hộp 5 − tổng các món",
           ("D1", "rice"): "451 − đĩa đen 167 − tổng các món"}


# ----------------------------------------------------------------- mask -> polygon
def load_cache(idx):
    z = np.load(os.path.join(CACHE, f"{idx:03d}.npz"))
    n = int(z["n"])
    if n == 0:
        return [], [], None
    H, W = (int(v) for v in z["shape"])
    masks = np.unpackbits(z["masks"], axis=-1, count=W).astype(bool)
    return list(masks), list(z["logits"].astype(np.float32)), (H, W)


def clean_mask(mk, min_area, hole_area):
    mk = mk.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    mk = keep[lab]
    H, W = mk.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((~mk).astype(np.uint8), connectivity=4)
    x, y, w, h, a = st[:, 0], st[:, 1], st[:, 2], st[:, 3], st[:, 4]
    border = (x == 0) | (y == 0) | (x + w == W) | (y + h == H)
    fill = (~border) & (a < hole_area)
    fill[0] = False
    return mk | fill[lab]


def resolve(idx, items, masks, logits, shape):
    """Mỗi pixel thuộc đúng 1 món: z cao thắng; cùng z thì logit SAM cao thắng; cơm (z=0) nằm dưới."""
    H, W = shape
    min_area = MIN_AREA_FRAC * H * W
    hole = HOLE_FRAC * H * W
    best = np.full((H, W), -1e9, np.float32)
    owner = np.full((H, W), -1, np.int32)
    for i, it in enumerate(items):
        z = it["z"] if it["z"] is not None else (0 if it["cls"] == "rice" else 1)
        mk = clean_mask(masks[i], min_area, hole)
        lg = cv2.resize(logits[i], (W, H), interpolation=cv2.INTER_LINEAR)
        s = z * 100.0 + np.clip(lg, -40, 40)
        upd = mk & (s > best)
        owner[upd] = i
        best[upd] = s[upd]
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    finals = []
    for i in range(len(items)):
        mk = (owner == i).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, kern)          # bỏ dải mỏng dính mép
        finals.append(clean_mask(mk.astype(bool), min_area, hole))
    return finals


def polys_of(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for g in geom.geoms:
            out += polys_of(g)
        return out
    return []


def mask_to_geom(mk, min_area):
    cnts, hier = cv2.findContours(mk.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return None
    hier = hier[0]
    parts = []
    for i, c in enumerate(cnts):
        if hier[i][3] != -1 or len(c) < 3:
            continue
        holes, j = [], hier[i][2]
        while j != -1:
            if len(cnts[j]) >= 3 and cv2.contourArea(cnts[j]) >= min_area:
                holes.append(cnts[j][:, 0, :] + 0.5)
            j = hier[j][0]
        p = Polygon(c[:, 0, :] + 0.5, holes)
        if not p.is_valid:
            p = make_valid(p)
        p = p.simplify(SIMPLIFY_PX, preserve_topology=True)
        parts += [q for q in polys_of(p) if q.area >= min_area]
    return unary_union(parts) if parts else None


def build_geoms(idx, items, masks, logits, shape):
    H, W = shape
    min_area = MIN_AREA_FRAC * H * W
    finals = resolve(idx, items, masks, logits, shape)
    geoms = [mask_to_geom(mk, min_area) for mk in finals]
    # Cắt hình học theo thứ tự ưu tiên để bảo đảm diện tích chồng lấn = 0
    order = sorted(range(len(items)),
                   key=lambda i: (-(items[i]["z"] if items[i]["z"] is not None else (0 if items[i]["cls"] == "rice" else 1)),
                                  geoms[i].area if geoms[i] is not None else 0))
    acc = None
    out = [None] * len(items)
    for i in order:
        g = geoms[i]
        if g is None:
            continue
        if acc is not None:
            g = g.difference(acc)
        ps = [p for p in polys_of(g) if p.area >= min_area]
        if not ps:
            continue
        out[i] = ps
        u = unary_union(ps)
        acc = u if acc is None else acc.union(u)
    return out, finals


def ring_with_bridges(p):
    """Polygon có lỗ -> 1 vòng duy nhất bằng cầu nối rộng 0 (cách YOLO/COCO polygon biểu diễn lỗ)."""
    ext = [tuple(c) for c in list(p.exterior.coords)[:-1]]
    for hole in p.interiors:
        h = [tuple(c) for c in list(hole.coords)[:-1]]
        E, Hh = np.array(ext), np.array(h)
        d = ((E[:, None, :] - Hh[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(d.argmin(), d.shape)
        ext = ext[:i + 1] + h[j:] + h[:j + 1] + ext[i:]
    return ext


# ----------------------------------------------------------------- màu + vẽ
def class_colors(classes):
    base = [(255, 140, 0), (255, 200, 0), (34, 197, 94), (230, 30, 60), (150, 70, 220), (0, 170, 255),
            (255, 90, 190), (120, 200, 30), (0, 200, 170), (200, 120, 60), (90, 110, 255), (255, 60, 20),
            (160, 160, 0), (0, 130, 90), (210, 0, 150), (110, 70, 30), (0, 90, 200), (255, 170, 120),
            (130, 0, 60), (80, 180, 255), (190, 190, 190), (60, 60, 160), (240, 230, 90), (150, 220, 160),
            (100, 40, 140), (255, 110, 110), (40, 140, 40), (220, 160, 255), (180, 90, 0), (0, 60, 60)]
    return {c: base[i % len(base)] for i, c in enumerate(classes)}


def render(idx, items, polys, weights, colors, path):
    im = Image.open(os.path.join(SRC, FILES[idx])).convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    W = im.width
    lw = max(3, W // 300)
    font = ImageFont.truetype("arialbd.ttf", max(16, W // 42))
    tags = []
    for it, ps in zip(items, polys):
        if not ps:
            continue
        col = colors[it["cls"]]
        for p in ps:
            ring = ring_with_bridges(p)
            d.polygon(ring, fill=col + (45,))
        for p in ps:
            for ringc in [p.exterior] + list(p.interiors):
                pts = list(ringc.coords)
                d.line(pts, fill=col + (255,), width=lw, joint="curve")
        top = min((pt for p in ps for pt in p.exterior.coords), key=lambda t: t[1])
        tags.append((it["cls"], top, col))
    im = Image.alpha_composite(im.convert("RGBA"), ov)
    d = ImageDraw.Draw(im)
    seen = set()
    for cls, (x, y), col in tags:
        txt = cls + (f" {weights[cls]}g" if cls in weights and cls not in seen else "")
        seen.add(cls)
        tw, th = d.textbbox((0, 0), txt, font=font)[2:]
        x0 = min(max(0, x - tw / 2), W - tw - 8)
        y0 = max(0, y - th - 14)
        d.rectangle([x0, y0, x0 + tw + 8, y0 + th + 8], fill=col + (255,))
        d.text((x0 + 4, y0 + 2), txt, fill=(255, 255, 255), font=font)
    im.convert("RGB").save(path, quality=88)


# ----------------------------------------------------------------- chạy
def process(idx, data):
    items = data[idx]["items"]
    if not items:
        return items, []
    f = os.path.join(CACHE, f"{idx:03d}.npz")
    if not os.path.exists(f) or str(np.load(f)["hash"]) != prompt_hash(data[idx]):
        raise RuntimeError(f"#{idx}: cache cũ, chạy segment.py trước")
    masks, logits, shape = load_cache(idx)
    polys, _ = build_geoms(idx, items, masks, logits, shape)
    return items, polys


def all_classes(data):
    seen = []
    for idx in sorted(data):
        for it in data[idx]["items"]:
            if it["cls"] not in seen:
                seen.append(it["cls"])
    if "rice" in seen:
        seen.remove("rice")
        seen.insert(0, "rice")
    return seen


def cmd_preview(idxs):
    data = parse_prompts()
    colors = class_colors(all_classes(data))
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    for idx in idxs:
        items, polys = process(idx, data)
        w, _, _ = weights_for(idx)
        render(idx, items, polys, w, colors, os.path.join(PREVIEW_DIR, f"{idx:03d}.jpg"))
        missing = [it["cls"] for it, p in zip(items, polys) if not p]
        print(f"#{idx}: ok" + (f"  (mất món: {missing})" if missing else ""))


def cmd_export():
    data = parse_prompts()
    classes = all_classes(data)
    cid = {c: i for i, c in enumerate(classes)}
    colors = class_colors(classes)
    readme = os.path.join(OUT, "README.md")                  # README viết tay: giữ lại khi xuất lại
    readme_text = open(readme, "rb").read() if os.path.exists(readme) else None
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for sub in ["images", "labels", "labelme", "masks", "preview"]:
        os.makedirs(os.path.join(OUT, sub))
    if readme_text is not None:
        with open(readme, "wb") as f:
            f.write(readme_text)

    coco = {"info": {"description": "Image_dataset - polygon món ăn + cân nặng (g)", "version": "1.0"},
            "images": [], "annotations": [], "categories": [{"id": i, "name": c} for i, c in enumerate(classes)]}
    rows, ann_id, overlap_report = [], 1, []
    for idx in range(len(FILES)):
        fname = FILES[idx]
        stem = os.path.splitext(fname)[0]
        shutil.copy2(os.path.join(SRC, fname), os.path.join(OUT, "images", fname))
        im = Image.open(os.path.join(SRC, fname))
        W, H = im.size
        entry = data.get(idx, {"items": [], "meal": None, "note": None})
        items, polys = process(idx, data) if idx in data else ([], [])
        weights, meal, kind = weights_for(idx)
        meal = meal or entry.get("meal")                      # bữa không cân vẫn giữ mã bữa
        kind = kind or ("full_meal" if items else None)
        coco["images"].append({"id": idx, "file_name": fname, "width": W, "height": H,
                               "meal": meal, "image_type": kind, "note": entry.get("note"),
                               "scale_reading_g": READINGS.get(idx)})
        yolo, shapes = [], []
        sem = np.zeros((H, W), np.uint8)
        per_class_area = {}
        all_ps = []
        for it, ps in zip(items, polys):
            if not ps:
                continue
            c = it["cls"]
            wg = weights.get(c)
            for p in ps:
                ring = ring_with_bridges(p)
                all_ps.append(p)
                yolo.append(f"{cid[c]} " + " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in ring))
                shapes.append({"label": c, "points": [[round(x, 2), round(y, 2)] for x, y in ring],
                               "group_id": None, "description": "" if wg is None else f"weight_g={wg}",
                               "shape_type": "polygon", "flags": {}})
                minx, miny, maxx, maxy = p.bounds
                coco["annotations"].append({
                    "id": ann_id, "image_id": idx, "category_id": cid[c], "iscrowd": 0,
                    "segmentation": [[round(v, 2) for xy in ring for v in xy]],
                    "area": round(p.area, 1), "bbox": [round(minx, 1), round(miny, 1), round(maxx - minx, 1), round(maxy - miny, 1)],
                    "attributes": {"weight_g": wg, "dish_weight_note": "cân nặng của cả món trong ảnh (mọi polygon cùng lớp)"}})
                ann_id += 1
                cv2.fillPoly(sem, [np.round(np.array(ring)).astype(np.int32)], cid[c] + 1)
                per_class_area[c] = per_class_area.get(c, 0) + p.area
        # kiểm tra chồng lấn
        ov = 0.0
        for a in range(len(all_ps)):
            for b in range(a + 1, len(all_ps)):
                if all_ps[a].intersects(all_ps[b]):
                    ov += all_ps[a].intersection(all_ps[b]).area
        overlap_report.append(ov)

        with open(os.path.join(OUT, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(yolo) + ("\n" if yolo else ""))
        with open(os.path.join(OUT, "labelme", stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": f"../images/{fname}",
                       "imageData": None, "imageHeight": H, "imageWidth": W}, f, ensure_ascii=False, indent=1)
        Image.fromarray(sem).save(os.path.join(OUT, "masks", stem + ".png"))
        render(idx, items, polys, weights, colors, os.path.join(OUT, "preview", f"{idx:03d}_{stem[:20]}.jpg"))

        for c in per_class_area:
            if c in weights:
                src = DERIVED.get((meal, c), "cân trực tiếp" if kind == "single_item" else "lấy từ ảnh cân lẻ cùng bữa")
                rows.append({"image_index": idx, "file_name": fname, "meal": meal, "image_type": kind,
                             "class": c, "weight_g": weights.get(c, ""), "weight_source": src,
                             "area_px": round(per_class_area[c]), "scale_reading_g": READINGS.get(idx, "")})
        for c in weights:
            if c not in per_class_area:
                print(f"  ! #{idx}: có cân nặng cho '{c}' nhưng ảnh không có polygon lớp này")

    with open(os.path.join(OUT, "annotations_coco.json"), "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    with open(os.path.join(OUT, "classes.txt"), "w") as f:
        f.write("\n".join(classes) + "\n")
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {OUT}\ntrain: images\nval: images\nnames:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")
    with open(os.path.join(OUT, "weights.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(OUT, "meals.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["meal", "class", "weight_g", "weight_source", "full_meal_images"])
        for m, (w, imgs) in MEALS.items():
            for c, g in w.items():
                wr.writerow([m, c, g, DERIVED.get((m, c), "cân trực tiếp (đã trừ bì)"), " ".join(map(str, imgs))])
    print(f"Xong: {len(FILES)} ảnh, {ann_id - 1} polygon, {len(classes)} lớp; "
          f"tổng diện tích chồng lấn = {sum(overlap_report):.2f} px²")


if __name__ == "__main__":
    if sys.argv[1] == "preview":
        cmd_preview([int(a) for a in sys.argv[2:]])
    elif sys.argv[1] == "export":
        cmd_export()
