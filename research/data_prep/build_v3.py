"""v3 = v2 + tập TEST khoá + biến thể nhãn "cơm gộp một polygon/đĩa".

- val GIỮ NGUYÊN 49 ảnh của v2 (điểm val so được với các lượt trước: baseline 0.505, aug 0.527)
- test cắt từ các nhóm buổi chụp đang ở TRAIN của v2, trừ Roboflow và nhóm khổng lồ.
  Không lấy từ val vì val đã dùng để ra quyết định "aug có tác dụng" và giả thuyết về cơm.
- mỗi lớp có trong test phải còn >= 1 instance ở train
- labels_rice1/: cùng ảnh, nhưng mọi polygon `rice` trong một ảnh được nối thành MỘT
  (ultralytics merge_multi_segment) — dùng cho cải tiến #2

Không sửa D:/food_dataset_v2.

    python build_v3.py            # in báo cáo
    python build_v3.py --write    # ghi D:/food_dataset_v3
"""
import os, csv, shutil, argparse, collections
import numpy as np
from ultralytics.data.converter import merge_multi_segment

SRC, OUT = r"D:/food_dataset_v2", r"D:/food_dataset_v3"
TEST_N, SEED = 37, 0
NAMES = [l.split(":", 1)[1].strip() for l in open(f"{SRC}/data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
RICE = NAMES.index("rice")


def rows_of(it):
    return [l.split() for l in open(f"{SRC}/labels/{it['split']}/{it['image'][:-4]}.txt") if l.strip()]


def merge_rice(rows):
    """Nối mọi polygon rice thành một. Toạ độ chuẩn hoá -> nối theo khoảng cách chuẩn hoá (đủ dùng để chọn điểm nối)."""
    rice = [np.array(r[1:], float) for r in rows if int(r[0]) == RICE]
    other = [" ".join(r) for r in rows if int(r[0]) != RICE]
    if len(rice) <= 1:
        return [" ".join(r) for r in rows], len(rice)
    pts = np.concatenate(merge_multi_segment([s.tolist() for s in rice]), axis=0)
    return other + [f"{RICE} " + " ".join(f"{v:.6f}" for v in pts.ravel())], len(rice)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); a = ap.parse_args()
    man = list(csv.DictReader(open(f"{SRC}/manifest.csv", encoding="utf-8-sig")))
    for it in man:
        it["rows"] = rows_of(it); it["cls"] = collections.Counter(NAMES[int(r[0])] for r in it["rows"])
    groups = collections.defaultdict(list)
    for it in man: groups[int(it["group"])].append(it)

    total = collections.Counter()
    for it in man:
        if it["split"] == "train": total.update(it["cls"])
    giant = max(groups, key=lambda g: len(groups[g]))
    cand = [g for g, v in groups.items()
            if all(it["split"] == "train" and it["source"] != "RF" for it in v) and g != giant]
    rng = np.random.default_rng(SEED)
    order = sorted(cand, key=lambda g: (len(groups[g]), rng.random()))
    test, n, tc = set(), 0, collections.Counter()
    for g in order:
        if n >= TEST_N: break
        gc = collections.Counter()
        for it in groups[g]: gc.update(it["cls"])
        if len(groups[g]) > TEST_N - n + 4: continue
        if all(total[c] - tc[c] - gc[c] >= 1 for c in gc):
            test.add(g); n += len(groups[g]); tc.update(gc)
    for it in man:
        it["new"] = "test" if int(it["group"]) in test else it["split"]

    cnt = collections.Counter(it["new"] for it in man)
    cls_in = {s: set().union(*[set(it["cls"]) for it in man if it["new"] == s]) for s in ("train", "val", "test")}
    print("chia:", dict(cnt), "| nhóm test:", len(test))
    print("lớp có ở test:", len(cls_in["test"]), "| lớp ở test mà không ở train:", sorted(cls_in["test"] - cls_in["train"]))
    multi = sum(1 for it in man if it["cls"]["rice"] > 1)
    print(f"ảnh có >1 polygon rice: {multi}/{len(man)}")
    if not a.write: return

    if os.path.exists(OUT): shutil.rmtree(OUT)
    merged_check = 0
    for it in man:
        s = it["new"]; stem = it["image"][:-4]
        for d in ("images", "labels", "labels_rice1"): os.makedirs(f"{OUT}/{d}/{s}", exist_ok=True)
        shutil.copy2(f"{SRC}/images/{it['split']}/{it['image']}", f"{OUT}/images/{s}/{it['image']}")
        shutil.copy2(f"{SRC}/labels/{it['split']}/{stem}.txt", f"{OUT}/labels/{s}/{stem}.txt")
        rows, k = merge_rice(it["rows"])
        assert sum(1 for r in rows if int(r.split()[0]) == RICE) == min(k, 1)
        merged_check += k > 1
        open(f"{OUT}/labels_rice1/{s}/{stem}.txt", "w").write("\n".join(rows) + "\n")
    with open(f"{OUT}/data.yaml", "w", encoding="utf-8") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
        for i, nm in enumerate(NAMES): f.write(f"  {i}: {nm}\n")
    with open(f"{OUT}/manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["image", "source", "split", "split_v2", "group", "classes"])
        for it in man: w.writerow([it["image"], it["source"], it["new"], it["split"], it["group"], it["classes"]])
    for fn in ("weights.csv", "class_counts.csv", "excluded.csv"):
        shutil.copy2(f"{SRC}/{fn}", f"{OUT}/{fn}")
    print("đã gộp rice ở", merged_check, "ảnh; ghi", OUT)


if __name__ == "__main__":
    main()
