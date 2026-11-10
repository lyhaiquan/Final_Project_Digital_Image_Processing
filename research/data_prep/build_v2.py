"""Gộp TOÀN BỘ ảnh đã gán nhãn trên máy thành một bộ YOLO-seg 47 lớp: D:/food_dataset_v2

Nguồn (thứ tự ưu tiên khi trùng ảnh, khử trùng theo md5 của ảnh GỐC sạch):
  B   D:/CVision/ảnh đã gán nhãn         118 ảnh LabelMe có gram. Ảnh trong folder là bản đã vẽ đè,
                                         ảnh sạch lấy từ D:/food-labeling/raw_zalo_376 (khớp ở v2_match_B.tsv)
  A   D:/food_yolo_seg (v1)              152 ảnh polygon SAM2, 40 lớp
  BN  bang_nhan/ảnh đã gán nhãn          64 ảnh LabelMe đã vẽ đè; chỉ 4 ảnh tìm được bản gốc (v2_match_bangnhan.tsv)
  RF  bang_nhan/Dataset/KQ (Roboflow)    49 ảnh COCO; chỉ giữ ảnh mà MỌI nhãn đều thuộc 47 lớp -> chỉ vào train

Không sửa file nào trong các folder nguồn.

    python build_v2.py            # in báo cáo
    python build_v2.py --write    # ghi D:/food_dataset_v2
"""
import os, sys, json, glob, csv, hashlib, collections, argparse, shutil
import numpy as np
from PIL import Image, ImageOps

HERE   = os.path.dirname(os.path.abspath(__file__))
B_DIR  = r"D:/CVision/ảnh đã gán nhãn"
RAW    = [r"D:/food-labeling/raw_zalo_376", r"D:/Image_dataset"]
A_DIR  = r"D:/food_yolo_seg"
BN_DIR = r"D:/food-labeling/bang_nhan/ảnh đã gán nhãn"
RF_DIR = r"D:/food-labeling/bang_nhan/Dataset/KQ"
OUT    = r"D:/food_dataset_v2"
VAL_FRAC, SEED = 0.2, 0

# Roboflow gõ khác tên cho cùng một món
RF_SYN = {"boil_cabbage": "cabbage", "boiled_cabbage": "cabbage",
          "stir_fried_morning_glory": "water_spinach"}


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def taxonomy():
    names = [l.split(":", 1)[1].strip() for l in open(os.path.join(A_DIR, "data.yaml"), encoding="utf-8")
             if l.strip()[:1].isdigit()]
    tax = dict(enumerate(names))
    for p in glob.glob(os.path.join(B_DIR, "file json", "*.json")):
        for k, v in json.load(open(p, encoding="utf-8"))["class_ids"].items():
            assert tax.get(v, k) == k, f"class_ids lệch: {k}={v} nhưng v1 có {tax[v]}"
            tax[v] = k
    assert sorted(tax) == list(range(len(tax)))
    return [tax[i] for i in range(len(tax))]


NAMES = taxonomy()
NID = {n: i for i, n in enumerate(NAMES)}
RAW_BY_NAME = {os.path.basename(p): p for d in RAW for p in glob.glob(os.path.join(d, "*.jpg"))}


def norm_rows(polys, W, H):
    """polys: [(class_name, [(x,y),...])] pixel -> dòng YOLO-seg, bỏ đa giác < 3 điểm."""
    rows, bad = [], 0
    for name, pts in polys:
        if len(pts) < 3: bad += 1; continue
        xy = " ".join(f"{min(max(x / W, 0), 1):.6f} {min(max(y / H, 0), 1):.6f}" for x, y in pts)
        rows.append(f"{NID[name]} {xy}")
    return rows, bad


def load_clean(p, W, H):
    """Ảnh gốc, xoay theo EXIF vào pixel; khớp đúng khung JSON (cùng tỉ lệ tuyệt đối)."""
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    if im.size != (W, H):
        assert im.width * H == im.height * W, f"{p}: {im.size} khác tỉ lệ {W}x{H}"
        im = im.resize((W, H), Image.LANCZOS)
    return im


def collect():
    items, dropped = [], []      # item: dict(key, src, md5, rows, img(fn), split_force, cls, weights, meta)

    # ---- B
    match = {r["stem"]: r["raw_file"] for r in csv.DictReader(open(os.path.join(HERE, "v2_match_B.tsv"), encoding="utf-8"), delimiter="\t")}
    for jp in sorted(glob.glob(os.path.join(B_DIR, "file json", "*.json")), key=lambda s: int(os.path.basename(s)[:-5])):
        stem = os.path.basename(jp)[:-5]; j = json.load(open(jp, encoding="utf-8"))
        if stem not in match:
            dropped.append(("B", stem, "có JSON nhưng không có ảnh nào (kể cả bản vẽ đè) để tìm ảnh gốc")); continue
        raw = RAW_BY_NAME[match[stem]]; W, H = j["imageWidth"], j["imageHeight"]
        polys = [(s["label"], [tuple(p) for p in s["points"]]) for s in j["shapes"]]
        rows, bad = norm_rows(polys, W, H)
        # gram -> những dòng polygon tạo nên món đó
        wts = []
        for key, g in j["food_weights_g"].items():
            cls, _, tag = key.partition("#")
            idx = [i for i, s in enumerate(j["shapes"]) if s["label"] == cls and
                   ((f"({tag})" in (s["description"] or "")) if tag else "(" not in (s["description"] or ""))]
            wts.append(dict(cls=cls, tag=tag, grams=g, rows=idx, how=j["weight_source"].get(key, "")))
        items.append(dict(key=f"B_{int(stem):03d}", src="B", num=int(stem), md5=md5(raw), orig=raw, W=W, H=H,
                          rows=rows, bad=bad, cls=frozenset(n for n, _ in polys), weights=wts,
                          wsig=tuple(sorted(j["food_weights_g"].items())), labelfile=jp))

    # ---- A
    for ip in sorted(glob.glob(os.path.join(A_DIR, "images", "*", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(A_DIR, "labels", os.path.basename(os.path.dirname(ip)), stem + ".txt")
        rows = [l.strip() for l in open(lp) if l.strip()]
        W, H = Image.open(ip).size
        items.append(dict(key=f"A_{stem[:13]}", src="A", md5=md5(ip), orig=ip, W=W, H=H, rows=rows, bad=0,
                          cls=frozenset(NAMES[int(r.split()[0])] for r in rows), weights=[], labelfile=lp,
                          stem=stem))

    # ---- BN: chỉ những ảnh tìm được gốc
    bn = {r["stem"]: r["raw_file"] for r in csv.DictReader(open(os.path.join(HERE, "v2_match_bangnhan.tsv"), encoding="utf-8"), delimiter="\t")}
    for jp in sorted(glob.glob(os.path.join(BN_DIR, "file json", "*.json")), key=lambda s: int(os.path.basename(s)[:-5])):
        stem = os.path.basename(jp)[:-5]
        if stem not in bn:
            dropped.append(("BN", stem, "ảnh đã bị vẽ đè, không tìm thấy ảnh gốc trên máy")); continue
        j = json.load(open(jp, encoding="utf-8")); raw = RAW_BY_NAME[bn[stem]]
        rows, bad = norm_rows([(s["label"], [tuple(p) for p in s["points"]]) for s in j["shapes"]], j["imageWidth"], j["imageHeight"])
        items.append(dict(key=f"BN_{int(stem):03d}", src="BN", md5=md5(raw), orig=raw, W=j["imageWidth"], H=j["imageHeight"],
                          rows=rows, bad=bad, cls=frozenset(s["label"] for s in j["shapes"]), weights=[], labelfile=jp))

    # ---- RF
    for sp in ("train", "valid", "test"):
        j = json.load(open(os.path.join(RF_DIR, sp, "_annotations.coco.json"), encoding="utf-8"))
        cat = {c["id"]: RF_SYN.get(c["name"], c["name"]) for c in j["categories"]}
        anns = collections.defaultdict(list)
        for a in j["annotations"]: anns[a["image_id"]].append(a)
        for im in j["images"]:
            names = {cat[a["category_id"]] for a in anns[im["id"]]}
            miss = sorted(n for n in names if n not in NID)
            if miss:
                dropped.append(("RF", im["file_name"], "có món ngoài 47 lớp: " + ", ".join(miss))); continue
            if not anns[im["id"]]:
                dropped.append(("RF", im["file_name"], "không có nhãn nào")); continue
            polys = [(cat[a["category_id"]], list(zip(a["segmentation"][0][0::2], a["segmentation"][0][1::2])))
                     for a in anns[im["id"]]]
            rows, bad = norm_rows(polys, im["width"], im["height"])
            p = os.path.join(RF_DIR, sp, im["file_name"])
            items.append(dict(key="RF_" + im["file_name"].split(".rf.")[0], src="RF", md5=md5(p), orig=p,
                              W=im["width"], H=im["height"], rows=rows, bad=bad, cls=frozenset(names),
                              weights=[], labelfile=p, force="train"))
    return items, dropped


def dedupe(items, dropped):
    prio = {"B": 0, "A": 1, "BN": 2, "RF": 3}
    by = collections.defaultdict(list)
    for it in items: by[it["md5"]].append(it)
    keep = []
    for v in by.values():
        v.sort(key=lambda it: (prio[it["src"]], it["key"]))
        keep.append(v[0])
        for it in v[1:]:
            dropped.append((it["src"], it["key"], f"trùng ảnh với {v[0]['key']} (giữ nhãn của {v[0]['src']})"))
        v[0]["dups"] = [it["key"] for it in v[1:]]
        # ảnh A bị thay bằng bản B vẫn phải kéo theo nhóm buổi chụp cũ của nó
        v[0]["a_alias"] = [it["stem"] for it in v if it["src"] == "A"]
    assert len({it["key"] for it in keep}) == len(keep), "trùng tên file đầu ra"
    return keep


class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def group(items):
    """Gộp vào cùng nhóm nếu có BẤT KỲ tín hiệu nào cho thấy cùng một khay. Gộp thừa an toàn hơn gộp thiếu."""
    uf = UF(len(items))
    def join(lst):
        for x in lst[1:]: uf.u(lst[0], x)
    bucket = collections.defaultdict(list)
    for i, it in enumerate(items):
        if it["src"] == "RF": continue
        bucket[("cls", it["cls"])].append(i)                         # trùng tập món
        if it.get("wsig"): bucket[("w", it["wsig"])].append(i)       # trùng bảng cân = cùng bữa
    for v in bucket.values(): join(v)
    # B: số thứ tự liền nhau = các góc chụp liên tiếp
    num = {it["num"]: i for i, it in enumerate(items) if it["src"] == "B"}
    for n, i in num.items():
        if n + 1 in num: uf.u(i, num[n + 1])
    # A: giữ nguyên nhóm buổi chụp đã suy ở v1
    a_files = sorted(os.listdir(r"D:/Image_dataset"))
    a_idx = {os.path.splitext(f)[0]: k for k, f in enumerate(a_files)}
    a_items = {st: i for i, it in enumerate(items) for st in it.get("a_alias", [])}
    for l in open(os.path.join(A_DIR, "split_groups.txt"), encoding="utf-8"):
        if l.startswith("#"): continue
        mem = [a_items[os.path.splitext(a_files[k])[0]] for k in json.loads(l.split("\t")[2])
               if os.path.splitext(a_files[k])[0] in a_items]
        join(mem)
    g = collections.defaultdict(list)
    for i, it in enumerate(items): g[uf.f(i)].append(i)
    return list(g.values())


def split(items, groups):
    rng = np.random.default_rng(SEED)
    cnt = lambda g: collections.Counter(NAMES[int(r.split()[0])] for i in g for r in items[i]["rows"])
    gc = [cnt(g) for g in groups]
    total = collections.Counter()
    for c in gc: total.update(c)
    pool = [k for k, g in enumerate(groups) if not any(items[i].get("force") for i in g)]
    target = round(sum(len(groups[k]) for k in pool) * VAL_FRAC)
    order = sorted(pool, key=lambda k: (len(groups[k]), rng.random()))
    val, nval, vc = set(), 0, collections.Counter()
    for k in order:
        if nval >= target: break
        if len(groups[k]) > target - nval + 5: continue
        if all(total[c] - vc[c] - gc[k][c] >= 1 for c in gc[k]):
            val.add(k); nval += len(groups[k]); vc.update(gc[k])
    for k, g in enumerate(groups):
        for i in g: items[i]["split"] = "val" if k in val else "train"; items[i]["group"] = k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    items, dropped = collect()
    n_in = collections.Counter(it["src"] for it in items)
    for it in [it for it in items if not it["rows"]]:
        # v1 có ảnh chụp hộp rỗng trên cân, ảnh sàn, ảnh nhoè. Ảnh có đồ ăn mà nhãn rỗng sẽ dạy model "đồ ăn là nền"
        dropped.append((it["src"], it["key"], "không có polygon nào (ảnh cân bì / ảnh hỏng)")); items.remove(it)
    items = dedupe(items, dropped)
    groups = group(items)
    split(items, groups)

    ci = collections.Counter(); cimg = collections.defaultdict(set)
    for it in items:
        for r in it["rows"]:
            c = NAMES[int(r.split()[0])]; ci[(c, it["split"])] += 1; cimg[c].add(it["key"])
    only_val = [n for n in NAMES if ci[(n, "val")] and not ci[(n, "train")]]
    only_tr = [n for n in NAMES if ci[(n, "train")] and not ci[(n, "val")]]
    absent = [n for n in NAMES if not cimg[n]]
    sp = collections.Counter((it["src"], it["split"]) for it in items)
    print("đọc vào:", dict(n_in), "| sau khử trùng:", len(items), "ảnh,", len(groups), "nhóm")
    print("train/val theo nguồn:", dict(sorted(sp.items())))
    print("lớp chỉ có ở val:", only_val, "| chỉ có ở train:", len(only_tr), "| không có ảnh nào:", absent)
    print("cỡ nhóm lớn nhất:", sorted((len(g) for g in groups), reverse=True)[:6])
    print("đa giác < 3 điểm bị bỏ:", sum(it["bad"] for it in items))
    print("loại:", collections.Counter(d[0] + ": " + d[2].split(":")[0].split(" (")[0] for d in dropped))
    if not a.write: return

    if os.path.exists(OUT): shutil.rmtree(OUT)
    for s in ("train", "val"):
        for d in ("images", "labels"): os.makedirs(os.path.join(OUT, d, s))
    for it in items:
        name = it["key"]
        if it["src"] in ("B", "BN"):
            load_clean(it["orig"], it["W"], it["H"]).save(os.path.join(OUT, "images", it["split"], name + ".jpg"), quality=95)
        else:
            im = Image.open(it["orig"])
            if im.getexif().get(0x0112, 1) != 1:   # A và RF: nếu còn tag xoay thì nướng vào pixel
                load_clean(it["orig"], it["W"], it["H"]).save(os.path.join(OUT, "images", it["split"], name + ".jpg"), quality=95)
            else:
                shutil.copy2(it["orig"], os.path.join(OUT, "images", it["split"], name + ".jpg"))
        with open(os.path.join(OUT, "labels", it["split"], name + ".txt"), "w") as f:
            f.write("\n".join(it["rows"]) + "\n")

    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES): f.write(f"  {i}: {n}\n")
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["image", "source", "split", "group", "n_polygons", "classes", "md5_original", "original", "label_from", "duplicates_dropped"])
        for it in sorted(items, key=lambda x: x["key"]):
            w.writerow([it["key"] + ".jpg", it["src"], it["split"], it["group"], len(it["rows"]), " ".join(sorted(it["cls"])),
                        it["md5"], it["orig"], it["labelfile"], " ".join(it.get("dups", []))])
    with open(os.path.join(OUT, "weights.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["image", "split", "class", "instance_tag", "grams", "label_rows", "weight_source"])
        for it in sorted(items, key=lambda x: x["key"]):
            for x in it["weights"]:
                w.writerow([it["key"] + ".jpg", it["split"], x["cls"], x["tag"], x["grams"], " ".join(map(str, x["rows"])), x["how"]])
    with open(os.path.join(OUT, "excluded.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["source", "item", "reason"]); w.writerows(dropped)
    with open(os.path.join(OUT, "class_counts.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "class", "images", "instances_train", "instances_val"])
        for i, n in enumerate(NAMES): w.writerow([i, n, len(cimg[n]), ci[(n, "train")], ci[(n, "val")]])
    print("Đã ghi", OUT)


if __name__ == "__main__":
    main()
