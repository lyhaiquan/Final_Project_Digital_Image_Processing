"""Xuất dữ liệu người ăn đã góp (UC-B06, FR-B17) ra định dạng YOLO-seg để nhóm rà lại nhãn và train tiếp.

    python -m backend.export_corrections out/corrections_2026-11

Kết quả:
    out/images/meal_<id>.jpg
    out/labels/meal_<id>.txt      mỗi dòng: <class id> x1 y1 x2 y2 ... (toạ độ chuẩn hoá 0-1)
    out/missing.csv               món người ăn thêm tay (không có vùng trên ảnh) -> cần khoanh tay
    out/data.yaml                 danh sách 45 lớp theo thứ tự id trong models/calories.csv
Chỉ dùng nội bộ: không mở thành API công khai.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from sqlalchemy import select

from .config import get_settings
from .db import Correction, make_engine, make_sessionmaker
from .services.catalog import Catalog


def export(out: Path, settings=None) -> int:
    s = settings or get_settings()
    cat = Catalog(s.models_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    sessions = make_sessionmaker(make_engine(s.database_url))
    n = 0
    with sessions() as db, open(out / "missing.csv", "w", newline="", encoding="utf-8") as fm:
        miss = csv.writer(fm)
        miss.writerow(["image", "cls", "name", "grams"])
        for c in db.scalars(select(Correction).order_by(Correction.id)):
            src = Path(c.image_path)
            if not src.exists():
                continue
            stem = src.stem
            shutil.copyfile(src, out / "images" / src.name)
            W, H = c.original_json["image"]["w"], c.original_json["image"]["h"]
            regions = {it["id"]: it for it in c.original_json["items"]}
            lines = []
            for it in c.corrected_json:
                reg = regions.get(it.get("region_id"))
                if reg is None:
                    miss.writerow([src.name, it["cls"], cat.name(it["cls"]), it["grams"]])
                    continue
                for poly in reg["polygons"]:
                    if len(poly) >= 3:
                        xy = " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in poly)
                        lines.append(f"{cat.rows[it['cls']]['id']} {xy}")
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            n += 1
    names = "\n".join(f"  {r['id']}: {r['cls']}" for r in sorted(cat.rows.values(), key=lambda r: r["id"]))
    (out / "data.yaml").write_text(f"path: .\ntrain: images\nval: images\nnames:\n{names}\n", encoding="utf-8")
    return n


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    print(f"Đã xuất {export(a.out)} ảnh vào {a.out}")
