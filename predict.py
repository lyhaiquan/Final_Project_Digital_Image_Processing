"""Chạy mô hình đã train trên ảnh khay cơm.

    python predict.py examples/                  # cả thư mục
    python predict.py anh1.jpg anh2.jpg --depth  # bật Depth-Anything cho cơm (lần đầu tải ~1.3 GB)
    python predict.py examples/ --no-dino        # chỉ YOLO, không phân loại lại bằng DINOv2

Kết quả ghi vào outputs/: <tên ảnh>_ketqua.jpg (ảnh có viền món) và <tên ảnh>.json (chi tiết từng món).
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2

from calo_khay import CaloKhay
from calo_khay.draw import draw

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Calo Khay: nhận diện món, ước lượng gram và calo từ ảnh khay cơm")
    ap.add_argument("inputs", nargs="+", help="ảnh hoặc thư mục ảnh")
    ap.add_argument("--depth", action="store_true", help="dùng Depth-Anything-V2 Large ước lượng cơm (nên có GPU)")
    ap.add_argument("--no-dino", action="store_true", help="tắt bước phân loại lại bằng DINOv2 (nhanh hơn, kém chính xác hơn)")
    ap.add_argument("--flag-conf", type=float, default=None,
                    help="độ tin cậy dưới ngưỡng này -> 'Cần xác nhận' (mặc định 0.75; 0.5 nếu --no-dino)")
    ap.add_argument("--device", default=None, help="vd: cpu, 0 (GPU số 0). Bỏ trống = tự chọn")
    ap.add_argument("--out", default="outputs", help="thư mục kết quả (mặc định outputs/)")
    a = ap.parse_args()

    files = []
    for p in a.inputs:
        files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXT)) if os.path.isdir(p) else [p]
    if not files:
        sys.exit("Không tìm thấy ảnh nào.")
    os.makedirs(a.out, exist_ok=True)

    print("Đang nạp mô hình..." + (" (kèm Depth-Anything)" if a.depth else ""))
    ck = CaloKhay(use_depth=a.depth, device=a.device, use_dino=not a.no_dino)
    for f in files:
        t = time.time()
        res = ck.analyze(f, flag_conf=a.flag_conf)
        stem = os.path.splitext(os.path.basename(f))[0]
        bgr = cv2.imread(f)
        cv2.imwrite(os.path.join(a.out, f"{stem}_ketqua.jpg"), draw(bgr, res))
        json.dump(res, open(os.path.join(a.out, f"{stem}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        print(f"\n== {os.path.basename(f)}  ({time.time() - t:.1f} s)")
        if res["warnings"]:
            print("   cảnh báo:", ", ".join(res["warnings"]))
        for it in res["items"]:
            alt = ", ".join(ck.name_vi.get(s, s) for s in it["suggestions"])
            flag = "  <- cần xác nhận" + (f" (có thể là: {alt})" if alt else "") \
                if it["needs_confirm"] else ""
            print(f"   {it['name']:<22} {it['grams']:>4} g {it['kcal']:>5} kcal  tin cậy {it['conf']:.2f}{flag}")
        tot = res["total"]
        print(f"   {'TỔNG':<22} {tot['grams']:>4} g {tot['kcal']:>5} kcal  (≈{tot['daily_pct']}% nhu cầu ngày, sai số ±{tot['error_pct']}%/món)")
    print(f"\nĐã lưu kết quả vào {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
