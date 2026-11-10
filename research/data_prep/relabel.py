"""Gộp lớp theo xác nhận của nhóm rồi đánh lại số thứ tự lớp: D:/food_dataset_v3 -> D:/food_dataset_v4

Giữ NGUYÊN cách chia train/val/test của v3 (chỉ đổi số lớp trong file nhãn), nên điểm v4 so được với v3.
Thêm cặp vào MERGE khi nhóm xác nhận thêm, rồi chạy lại.

    python relabel.py
"""
import os, csv, shutil, glob, collections

SRC, OUT = r"D:/food_dataset_v3", r"D:/food_dataset_v4"
MERGE = {
    "fried_pork_roll": "roasted_cinnamon_pork",   # 09/11: "tất cả fried_pork_roll là chả quế"
    "lemongrass_chili_chicken": "bell_pepper_chicken",   # 10/11: "gà xào ớt chuông và gà xào sả ớt là một"
}
# Sửa nhãn TỪNG ẢNH: (ảnh, lớp cũ) -> lớp mới, áp cho mọi polygon lớp đó trong ảnh (cả labels_rice1 và csv)
FIX = {
    # 10/11: "nó là gà chua ngọt" (B_169). B_161/B_168 là cùng khay, cùng miếng 49 g (cân 165.jpg)
    ("B_161.jpg", "fried_chicken"): "sweet_sour_chicken",
    ("B_168.jpg", "fried_chicken"): "sweet_sour_chicken",
    ("B_169.jpg", "fried_chicken"): "sweet_sour_chicken",
    # 10/11: khay B_170/171/173 (miếng 51 g, cân 179.jpg) cũng là gà chua ngọt
    ("B_170.jpg", "fried_chicken"): "sweet_sour_chicken",
    ("B_171.jpg", "fried_chicken"): "sweet_sour_chicken",
    ("B_173.jpg", "fried_chicken"): "sweet_sour_chicken",
}

old = [l.split(":", 1)[1].strip() for l in open(f"{SRC}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
new = [n for n in old if n not in MERGE]
remap = {i: new.index(MERGE.get(n, n)) for i, n in enumerate(old)}


def main():
    if os.path.exists(OUT): shutil.rmtree(OUT)
    shutil.copytree(f"{SRC}/images", f"{OUT}/images")
    cnt = collections.Counter()
    for tree in ("labels", "labels_rice1"):
        for p in glob.glob(f"{SRC}/{tree}/*/*.txt"):
            q = p.replace(SRC, OUT, 1); os.makedirs(os.path.dirname(q), exist_ok=True)
            rows = []; im = os.path.basename(p)[:-4] + ".jpg"
            for l in open(p):
                r = l.split()
                if not r: continue
                n = MERGE.get(old[int(r[0])], old[int(r[0])]); n = FIX.get((im, n), n)
                c = new.index(n); rows.append(" ".join([str(c)] + r[1:]))
                if tree == "labels": cnt[new[c]] += 1
            open(q, "w").write("\n".join(rows) + "\n")
    with open(f"{OUT}/data.yaml", "w", encoding="utf-8") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for i, n in enumerate(new): f.write(f"  {i}: {n}\n")
    # các file csv tham chiếu theo TÊN lớp -> đổi tên
    for fn in ("manifest.csv", "weights.csv"):
        rows = list(csv.reader(open(f"{SRC}/{fn}", encoding="utf-8-sig")))
        for r in rows[1:]:
            for k, v in enumerate(r):
                r[k] = " ".join(FIX.get((r[0], MERGE.get(t, t)), MERGE.get(t, t)) for t in v.split(" ")) if v else v
        csv.writer(open(f"{OUT}/{fn}", "w", newline="", encoding="utf-8-sig")).writerows(rows)
    shutil.copy2(f"{SRC}/excluded.csv", f"{OUT}/excluded.csv")
    print("sửa từng ảnh:", len(FIX), "| sweet_sour_chicken:", cnt["sweet_sour_chicken"], "polygon")
    print(f"{len(old)} -> {len(new)} lớp | gộp: {MERGE} | {new[remap[old.index(next(iter(MERGE)))]]}: {cnt[next(iter(MERGE.values()))]} polygon")


if __name__ == "__main__":
    main()
