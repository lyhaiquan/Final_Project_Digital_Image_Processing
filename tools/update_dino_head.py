"""Cập nhật đầu phân loại DINOv2 (models/dino_head.npz) từ ảnh người ăn đã góp. Không train lại YOLO.

Dữ liệu học = 1.654 vùng nhãn tay gốc (đặc trưng lưu sẵn ở models/dino_feats_base.npz, nên KHÔNG cần bộ dữ liệu gốc)
            + các vùng người ăn đã xác nhận / sửa tên (lấy từ bảng `corrections` của backend).
Chỉ lấy vùng đáng tin: bỏ món thêm tay (không có vùng trên ảnh) và món người ăn chưa xác nhận (confirmed = false).

Chốt chặn trước khi ghi đè:
  1. Để riêng ~25% số ẢNH góp (theo ảnh, không theo vùng) làm tập kiểm tra.
  2. Fit đầu thử trên (gốc + 75% ảnh góp), so với đầu đang dùng trên tập kiểm tra.
     Đầu thử phải KHÔNG kém hơn trên tập kiểm tra, và không "quên" dữ liệu gốc (độ chính xác trên vùng gốc giảm <= 2 điểm).
  3. Qua chốt -> fit lại trên TOÀN BỘ (gốc + mọi ảnh góp), sao lưu đầu cũ, ghi đè models/dino_head.npz.
Mặc định chỉ chạy thử và in kết quả; thêm --apply mới ghi file. Ghi xong phải KHỞI ĐỘNG LẠI backend để nạp đầu mới.

    python tools/update_dino_head.py                       # đọc DB theo .env của backend (CALO_DATABASE_URL), chạy thử
    python tools/update_dino_head.py --apply               # qua chốt thì ghi đè
    python tools/update_dino_head.py --labels out/corr     # dùng thư mục đã xuất bằng backend.export_corrections
                                                           # (sau khi nhóm rà lại nhãn bằng tay) thay vì đọc DB
    python tools/update_dino_head.py --build-base          # (chỉ nhóm train) tạo lại dino_feats_base.npz từ D:/food_dataset_v4

Cần thêm: pip install scikit-learn
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
HEAD, BASE = MODELS / "dino_head.npz", MODELS / "dino_feats_base.npz"
C_REG = 100            # như tools/export_dino_head.py (chọn trên val)
MIN_REGIONS = 20       # ít hơn số vùng góp này thì chưa cập nhật
MAX_FORGET = 2.0       # độ chính xác trên vùng gốc được phép giảm tối đa (điểm %)


# ----------------------------------------------------------------------------- đọc dữ liệu góp
def regions_from_db(database_url: str | None):
    """-> [(ảnh, mask polygon list, mã món)] từ bảng corrections."""
    from sqlalchemy import select
    from backend.config import get_settings
    from backend.db import Correction, make_engine, make_sessionmaker
    url = database_url or get_settings().database_url
    out, skipped = [], dict(them_tay=0, chua_xac_nhan=0, mat_anh=0)
    with make_sessionmaker(make_engine(url))() as db:
        for c in db.scalars(select(Correction).order_by(Correction.id)):
            if not Path(c.image_path).exists():
                skipped["mat_anh"] += 1; continue
            regions = {it["id"]: it for it in c.original_json["items"]}
            for it in c.corrected_json:
                reg = regions.get(it.get("region_id"))
                if reg is None or it.get("source") == "manual":
                    skipped["them_tay"] += 1; continue
                if it.get("confirmed") is False:
                    skipped["chua_xac_nhan"] += 1; continue
                polys = [p for p in reg["polygons"] if len(p) >= 3]
                if polys:
                    out.append((c.image_path, polys, it["cls"]))
    return out, skipped


def regions_from_labels(folder: Path, names: list[str]):
    """Thư mục định dạng YOLO-seg (images/, labels/) do backend.export_corrections xuất ra."""
    out = []
    for lbl in sorted((folder / "labels").glob("*.txt")):
        img = next((p for p in (folder / "images").glob(lbl.stem + ".*")), None)
        if img is None:
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        for line in lbl.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) >= 7:
                xy = (np.array(t[1:], float).reshape(-1, 2) * [w, h]).round().astype(int).tolist()
                out.append((str(img), [xy], names[int(t[0])]))
    return out, {}


def featurize(rc, regions):
    """Gom theo ảnh để mỗi ảnh chỉ đọc một lần. -> X [N, 768], y [N] (mã món), img [N]"""
    by_img: dict[str, list] = {}
    for img, polys, cls in regions:
        by_img.setdefault(img, []).append((polys, cls))
    X, Y, I = [], [], []
    for img, items in by_img.items():
        bgr = cv2.imread(img); H, W = bgr.shape[:2]
        masks = []
        for polys, cls in items:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.array(p, np.int32) for p in polys], 1)
            if m.any():
                masks.append(m > 0); Y.append(cls); I.append(img)
        X.append(rc.features(bgr, masks))
    return (np.concatenate(X) if X else np.zeros((0, 768), np.float32)), np.array(Y), np.array(I)


# ----------------------------------------------------------------------------- đầu phân loại
def fit(X, y, names, w=None):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C_REG, max_iter=5000).fit(X, [names.index(c) for c in y], sample_weight=w)
    coef = np.zeros((len(names), X.shape[1]), np.float32); b = np.full(len(names), -1e4, np.float32)
    coef[clf.classes_], b[clf.classes_] = clf.coef_, clf.intercept_       # lớp không có mẫu -> không bao giờ được chọn
    return dict(coef=coef, intercept=b, names=np.array(names))


def predict(head, X):
    return np.asarray(head["names"])[np.argmax(X @ np.asarray(head["coef"]).T + np.asarray(head["intercept"]), 1)]


def acc(head, X, y):
    return 100 * float((predict(head, X) == y).mean()) if len(y) else float("nan")


# ----------------------------------------------------------------------------- tạo file đặc trưng gốc
def build_base(rc, dataset: Path):
    names = [l.split(":", 1)[1].strip() for l in open(dataset / "data.yaml", encoding="utf-8") if l.strip()[:1].isdigit()]
    import pandas as pd
    man = pd.read_csv(dataset / "manifest.csv")
    regions = []
    for im, sp in man[man.split.isin(["train", "val"])][["image", "split"]].itertuples(index=False):
        p = dataset / "images" / sp / im
        h, w = cv2.imread(str(p)).shape[:2]
        for line in open(dataset / "labels" / sp / f"{im[:-4]}.txt", encoding="utf-8"):
            t = line.split()
            if t:
                regions.append((str(p), [(np.array(t[1:], float).reshape(-1, 2) * [w, h]).round().astype(int).tolist()], names[int(t[0])]))
    X, y, img = featurize(rc, regions)
    np.savez_compressed(BASE, X=X.astype(np.float16), y=y, image=np.array([Path(i).name for i in img]))
    print(f"Đã lưu {len(y)} vùng gốc -> {BASE} ({BASE.stat().st_size / 1e6:.1f} MB)")
    # kiểm tra: fit lại trên đặc trưng vừa lưu phải cho đầu dự đoán giống đầu đang dùng
    old = dict(np.load(HEAD))
    Xb = X.astype(np.float16).astype(np.float32)
    agree = 100 * float((predict(fit(Xb, y, [str(n) for n in old["names"]]), Xb) == predict(old, Xb)).mean())
    print(f"Kiểm tra: đầu fit lại từ file này trùng dự đoán với dino_head.npz trên {agree:.1f}% vùng gốc")


# ----------------------------------------------------------------------------- chính
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="qua chốt chặn thì ghi đè models/dino_head.npz")
    ap.add_argument("--labels", type=Path, help="thư mục do backend.export_corrections xuất (đã rà nhãn) thay vì đọc DB")
    ap.add_argument("--database-url", help="bỏ trống = CALO_DATABASE_URL trong .env")
    ap.add_argument("--weight", type=float, default=1.0, help="trọng số mỗi vùng góp so với vùng nhãn tay gốc (mặc định 1)")
    ap.add_argument("--holdout", type=float, default=0.25, help="tỉ lệ ảnh góp để riêng kiểm tra (mặc định 0.25)")
    ap.add_argument("--device", default=None, help="cpu, 0... bỏ trống = tự chọn")
    ap.add_argument("--build-base", type=Path, nargs="?", const=Path("D:/food_dataset_v4"), help=argparse.SUPPRESS)
    a = ap.parse_args()

    from calo_khay.pipeline import RegionClassifier
    rc = RegionClassifier(str(MODELS), a.device, load_head=False)
    if a.build_base:
        return build_base(rc, a.build_base)

    old = dict(np.load(HEAD)); names = [str(n) for n in old["names"]]
    base = np.load(BASE)
    Xb, yb = base["X"].astype(np.float32), base["y"].astype(str)

    regions, skipped = regions_from_labels(a.labels, names) if a.labels else regions_from_db(a.database_url)
    regions = [r for r in regions if r[2] in names]
    Xc, yc, ic = featurize(rc, regions)
    imgs = sorted(set(ic))
    print(f"Vùng góp dùng được: {len(yc)} vùng / {len(imgs)} ảnh" + (f" | bỏ qua: {skipped}" if skipped else ""))
    if len(yc) < MIN_REGIONS or len(imgs) < 4:
        print(f"Chưa đủ dữ liệu (cần >= {MIN_REGIONS} vùng và >= 4 ảnh). Giữ nguyên đầu hiện tại.")
        return

    # 1-2) chốt chặn: để riêng một phần ẢNH góp
    rng = np.random.default_rng(0)
    test_imgs = set(rng.choice(imgs, max(1, round(a.holdout * len(imgs))), replace=False))
    te = np.isin(ic, list(test_imgs))
    w = np.r_[np.ones(len(yb)), np.full(int((~te).sum()), a.weight)]
    cand = fit(np.vstack([Xb, Xc[~te]]), np.r_[yb, yc[~te]], names, w)
    r = dict(gop_kiem_tra_cu=acc(old, Xc[te], yc[te]), gop_kiem_tra_moi=acc(cand, Xc[te], yc[te]),
             goc_cu=acc(old, Xb, yb), goc_moi=acc(cand, Xb, yb), n_kiem_tra=int(te.sum()))
    print(f"Tập kiểm tra ({r['n_kiem_tra']} vùng góp, {len(test_imgs)} ảnh): đầu cũ {r['gop_kiem_tra_cu']:.1f}% -> đầu mới {r['gop_kiem_tra_moi']:.1f}%")
    print(f"Vùng nhãn tay gốc ({len(yb)} vùng):          đầu cũ {r['goc_cu']:.1f}% -> đầu mới {r['goc_moi']:.1f}%")
    ok = r["gop_kiem_tra_moi"] >= r["gop_kiem_tra_cu"] and r["goc_cu"] - r["goc_moi"] <= MAX_FORGET
    print("Qua chốt chặn." if ok else "KHÔNG qua chốt chặn: đầu mới không tốt hơn hoặc quên dữ liệu gốc. Giữ nguyên.")
    changed = {f"{a_} -> {b_}": int(n) for (a_, b_), n in zip(*np.unique(
        np.c_[predict(old, Xc), yc][predict(old, Xc) != yc], axis=0, return_counts=True))} if len(yc) else {}
    if changed:
        print("Các vùng đầu cũ đoán sai (đoán -> người ăn sửa):", dict(sorted(changed.items(), key=lambda x: -x[1])[:10]))
    if not ok or not a.apply:
        if ok:
            print("Chạy lại với --apply để ghi đè.")
        return

    # 3) fit trên toàn bộ, sao lưu, ghi đè
    final = fit(np.vstack([Xb, Xc]), np.r_[yb, yc], names, np.r_[np.ones(len(yb)), np.full(len(yc), a.weight)])
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MODELS / "dino_head_backup" / f"dino_head_{stamp}.npz"
    backup.parent.mkdir(exist_ok=True)
    shutil.copyfile(HEAD, backup)
    np.savez(HEAD, **final)
    log = MODELS / "dino_head_backup" / "history.jsonl"
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(time=stamp, backup=backup.name, n_base=len(yb), n_gop=len(yc), n_anh_gop=len(imgs),
                                weight=a.weight, **{k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}),
                           ensure_ascii=False) + "\n")
    print(f"Đã ghi {HEAD.name} (bản cũ: {backup.relative_to(ROOT)}). Khởi động lại backend để nạp đầu mới.")
    print(f"Muốn quay lại bản cũ: copy {backup.relative_to(ROOT)} đè lên models/{HEAD.name}")


if __name__ == "__main__":
    main()
