"""Chia train/val theo BUỔI CHỤP (không theo ảnh) rồi đóng gói cho Kaggle.

Vì 81/152 ảnh chưa được gán mã bữa, nhóm buổi chụp được suy lại một cách
BẢO THỦ bằng union-find trên 3 tín hiệu:
  1. cùng mã bữa trong weights_config.MEALS / IMAGE_ONLY / IMAGE_MEAL_EXTRA
  2. trùng tập lớp món trong file nhãn
  3. ảnh giống nhau về pixel (cosine >= 0.92 trên thumbnail 32x32 + histogram)

Gộp thừa thì chỉ mất chút hiệu quả dữ liệu; gộp thiếu thì rò rỉ train->val.
Nên mọi tín hiệu đều dùng để GỘP, không bao giờ để tách.

    python make_split.py            # in báo cáo, không ghi gì
    python make_split.py --write    # ghi D:/food_yolo_seg (cấu trúc Kaggle)
"""
import os, sys, shutil, collections, argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import weights_config as W

SRC   = r"D:\Image_dataset"
LAB   = r"D:\Image_dataset_labeled"
OUT   = r"D:\food_yolo_seg"
VAL_FRAC = 0.22
SEED  = 0

FILES  = sorted(os.listdir(SRC))
N      = len(FILES)
NAMES  = [l.strip() for l in open(os.path.join(LAB, "classes.txt")) if l.strip()]


def label_path(i):
    return os.path.join(LAB, "labels", os.path.splitext(FILES[i])[0] + ".txt")


def class_set(i):
    p = label_path(i)
    if not os.path.exists(p):
        return frozenset()
    return frozenset(NAMES[int(l.split()[0])] for l in open(p) if l.strip())


def pixel_feats():
    out = []
    for n in FILES:
        a = np.asarray(Image.open(os.path.join(SRC, n)).convert("RGB").resize((32, 32)), np.float32) / 255.0
        h = np.concatenate([np.histogram(a[:, :, c], 16, (0, 1))[0] for c in range(3)])
        out.append(np.concatenate([a.ravel(), h / h.sum() * 20]))
    X = np.stack(out)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b): self.p[self.find(a)] = self.find(b)


def build_groups():
    uf = UF(N)
    # (1) mã bữa đã biết
    meal_of = {}
    for m, (_, imgs) in W.MEALS.items():
        for i in imgs: meal_of[i] = m
    for i, (m, _) in W.IMAGE_ONLY.items(): meal_of[i] = m
    for i, m in W.IMAGE_MEAL_EXTRA.items(): meal_of[i] = m
    by_meal = collections.defaultdict(list)
    for i, m in meal_of.items(): by_meal[m].append(i)
    for v in by_meal.values():
        for x in v[1:]: uf.union(v[0], x)
    # (2) trùng tập lớp món (bỏ qua tập rỗng)
    by_set = collections.defaultdict(list)
    for i in range(N):
        s = class_set(i)
        if s: by_set[s].append(i)
    for v in by_set.values():
        for x in v[1:]: uf.union(v[0], x)
    # (3) ĐÃ BỎ: cosine trên thumbnail. Mọi ảnh đều là khay cơm trên nền giống nhau
    # nên single-linkage chuỗi hoá cả 152 ảnh thành 1 nhóm ở mọi ngưỡng < 0.97,
    # còn từ 0.97 trở lên thì không gộp thêm gì so với (1)+(2). Vô dụng ở cả hai đầu.
    g = collections.defaultdict(list)
    for i in range(N): g[uf.find(i)].append(i)
    return list(g.values()), meal_of


def split(groups):
    """Chọn nhóm cho val sao cho: (a) ~VAL_FRAC ảnh, (b) không lớp nào chỉ nằm ở val.

    Điều kiện (b) quan trọng: lớp chỉ có trong val thì model chưa bao giờ học nó,
    AP của nó luôn 0 và kéo mAP xuống một cách vô nghĩa.
    """
    rng = np.random.default_rng(SEED)
    gc = [per_group_count(g) for g in groups]
    total = collections.Counter()
    for c in gc: total.update(c)

    target = int(round(N * VAL_FRAC))
    order = sorted(range(len(groups)), key=lambda k: (len(groups[k]), rng.random()))
    val, nval, vcount = set(), 0, collections.Counter()
    for k in order:
        if nval >= target:
            break
        # thu dua nhom k vao val: moi lop trong k phai con it nhat 1 instance o train
        if all(total[c] - vcount[c] - gc[k][c] >= 1 for c in gc[k]):
            val.add(k); nval += len(groups[k]); vcount.update(gc[k])
    tr = [i for k, g in enumerate(groups) if k not in val for i in g]
    va = [i for k, g in enumerate(groups) if k in val for i in g]
    return sorted(tr), sorted(va)


def per_group_count(g):
    c = collections.Counter()
    for i in g:
        p = label_path(i)
        if os.path.exists(p):
            for l in open(p):
                if l.strip(): c[NAMES[int(l.split()[0])]] += 1
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    groups, meal_of = build_groups()
    tr, va = split(groups)

    ok = sum(len({next(k for k, g in enumerate(groups) if i in g) for i in im}) == 1
             for _, (_, im) in W.MEALS.items())
    print(f"nhom buoi chup suy ra : {len(groups)}  (7 bua da biet: {ok}/7 nguyen khoi)")
    print(f"train {len(tr)} anh | val {len(va)} anh  ({len(va)/N:.0%})")

    def per(idxs):
        c = collections.Counter()
        for i in idxs:
            p = label_path(i)
            if os.path.exists(p):
                for l in open(p):
                    if l.strip(): c[NAMES[int(l.split()[0])]] += 1
        return c
    ctr, cva = per(tr), per(va)
    only_tr = [n for n in NAMES if ctr[n] and not cva[n]]
    only_va = [n for n in NAMES if cva[n] and not ctr[n]]
    print(f"lop chi co trong train (val khong cham diem duoc): {len(only_tr)}")
    print(f"lop chi co trong val   (model chua bao gio hoc)   : {len(only_va)}")
    if only_va: print("   ->", only_va)

    if not a.write:
        print("\n(chay lai voi --write de ghi ra", OUT + ")")
        return

    for s in ("train", "val"):
        for d in ("images", "labels"):
            os.makedirs(os.path.join(OUT, d, s), exist_ok=True)
    for s, idxs in (("train", tr), ("val", va)):
        for i in idxs:
            stem = os.path.splitext(FILES[i])[0]
            shutil.copy2(os.path.join(SRC, FILES[i]), os.path.join(OUT, "images", s, FILES[i]))
            src_l = label_path(i); dst_l = os.path.join(OUT, "labels", s, stem + ".txt")
            if os.path.exists(src_l): shutil.copy2(src_l, dst_l)
            else: open(dst_l, "w").close()
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: .\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES): f.write(f"  {i}: {n}\n")
    with open(os.path.join(OUT, "split_groups.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {len(groups)} nhom buoi chup suy ra; val = nhom nao\n")
        for k, g in enumerate(groups):
            f.write(f"group{k:03d}\t{'val' if all(i in va for i in g) else 'train'}\t{g}\n")
    print(f"\nDa ghi {OUT}")


if __name__ == "__main__":
    main()
