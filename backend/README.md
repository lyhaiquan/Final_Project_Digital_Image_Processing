# Backend Calo Khay

FastAPI bọc pipeline `calo_khay.CaloKhay` thành API JSON, lưu nhật ký bữa ăn bằng **MySQL**.
Đặc tả: `docs/nghiep_vu.pdf` (mục 2–3). Tài liệu API tự sinh: <http://localhost:8000/docs>.

## 1. Chạy

```bash
pip install -r requirements-backend.txt      # + thư viện mô hình theo README gốc (ultralytics, torch, transformers)
cp .env.example .env                         # sửa nếu cần
docker compose up -d                         # MySQL 8.4: user calo / pass calo / db calo_khay
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

- Bảng MySQL tự tạo lúc khởi động (`scans`, `meals`, `meal_items`, `corrections`).
- Model nạp ở nền + chạy một ảnh làm nóng; trong lúc đó `/api/health` trả **503 `loading`**, `/api/scan` trả 503 `model_loading`.
- Chưa có model/GPU, hoặc chỉ cần API cho frontend: đặt `CALO_ENGINE=mock` → `/api/scan` trả `backend/mock/scan.json`
  (polygon co giãn theo ảnh gửi lên).
- Không có Docker: cài MySQL 8 rồi tạo DB `calo_khay` (utf8mb4) và sửa `CALO_DATABASE_URL`.

Test (SQLite + mock, không cần MySQL/GPU): `python -m pytest backend/tests -q`

## 2. Quyết định đã chốt (bổ sung/sửa SRS)

| # | Nội dung | Chốt |
|---|---|---|
| 1 | Ngưỡng "Cần xác nhận" | **0.75** khi có DINOv2 (0.5 khi tắt), đổi bằng `CALO_FLAG_CONF` hoặc query `flag_conf` |
| 2 | DINOv2 phân loại lại vùng | **Bật** (`CALO_USE_DINO`) |
| 3 | Độ sâu cho cơm | **Tắt** mặc định; bật `CALO_DEPTH_ENABLED=true` khi demo trên laptop GPU |
| 4 | Server demo | Laptop GPU của nhóm |
| 5–6 | Giữ kết quả quét + ảnh tạm | **24 giờ**, job dọn mỗi giờ |
| — | CSDL | **MySQL** thay SQLite |
| 7 | Định danh thiết bị | Header **`X-Device-Id`** (UUID v4 frontend tạo, lưu localStorage) |
| 8 | Thời gian | `eaten_at` gửi ISO 8601 có offset; DB lưu UTC; `?date=` tính theo giờ Việt Nam; trả về `+07:00` |
| 9 | Định dạng lỗi | `{"error": "<mã>", "message": "<câu tiếng Việt>"}` |
| 10 | Chọn suất cơm / chỉnh gram | Gửi `grams` trong item (giữ `source` như cũ) |
| 11 | Món chưa xác nhận | Item có `confirmed: false` |
| 12 | Ảnh thu nhỏ trong Nhật ký | **Frontend tự giữ** (IndexedDB); backend không trả ảnh |
| 13 | Frontend | Dev chạy riêng (CORS `localhost:5173`); demo: build rồi đặt `CALO_FRONTEND_DIST` |

## 3. API tóm tắt (chi tiết ở `/docs`)

| Endpoint | Ghi chú |
|---|---|
| `POST /api/scan` | multipart `image`; query `depth`, `flag_conf`; header `X-Device-Id` (tuỳ chọn). Trả kết quả pipeline + `scan_id`, `model_version`, `timing_ms`. Cảnh báo: `no_food`, `no_plate`, `depth_plane_failed`, `depth_unavailable`, `blurry` |
| `POST /api/recalc` | `{scan_id?, items: [{region_id?, cls, source, area_ratio?, grams?, orig_cls?, conf?, confirmed?}]}` |
| `POST /api/meals` | cần `X-Device-Id`. `{scan_id?, items, eaten_at, share_corrections}` → `201 {meal_id, total, correction_saved}` |
| `GET /api/meals?date=YYYY-MM-DD` | bữa trong ngày + `total_kcal`, `daily_pct` |
| `DELETE /api/meals/{id}` | 204; xoá luôn ảnh đã góp (nếu có) |
| `GET /api/classes` | 45 món: `cls, name, kcal_per_100g, default_g, source, suggestions` |
| `GET /api/benchmark` | `models/benchmark.json` (sinh bằng `python tools/build_benchmark_json.py`) |
| `GET /api/health` | 200 khi sẵn sàng, 503 khi đang nạp / lỗi |

**Cách `/api/recalc` (và `/api/meals`) tính gram cho từng item:**
1. Có `grams` → lấy đúng con số đó (suất cơm 150/227/300, chỉnh tay).
2. `source = manual` → khẩu phần trung bình của món (`default_g`).
3. `model` / `renamed` → công thức của món (cuối cùng) trên `area_ratio` của vùng; không có `area_ratio` (không thấy đĩa) → khẩu phần TB.
   Cơm lấy thể tích từ lần quét nếu đã bật độ sâu.

kcal luôn do backend tính lại (`gram × kcal/100 g ÷ 100`), không tin số frontend gửi.

**Luồng frontend gợi ý:** scan → giữ `items` (kèm `id` → gửi lại là `region_id`, `area_ratio`, `conf`) → mỗi lần sửa gọi
`/recalc` với toàn bộ danh sách → lưu bằng `/meals` với danh sách cuối.

## 4. Mã lỗi

| HTTP | `error` |
|---|---|
| 400 | `invalid_image`, `image_too_small`, `missing_device_id` |
| 404 | `scan_expired`, `meal_not_found`, `benchmark_missing` |
| 413 | `image_too_large` |
| 422 | `unknown_class`, `validation_error` (kèm `details`) |
| 503 | `busy` (hơn 8 ảnh chờ), `model_loading` — có header `Retry-After` |

## 5. Vận hành (nhóm phát triển)

- **Thay model / bảng calo:** thay file trong `models/` rồi khởi động lại (đổi `calories.csv` → `/classes` và kcal đổi theo).
- **Xuất ảnh người ăn đã góp** (định dạng YOLO-seg): `python -m backend.export_corrections out/corrections`
  → `images/`, `labels/`, `missing.csv` (món thêm tay cần khoanh), `data.yaml`.
- Log mỗi lần quét: số món, cảnh báo, thời gian từng bước, `model_version`; không log ảnh.

## 6. Cấu trúc

```
backend/
  app.py               tạo app, lifespan (nạp model, tạo bảng, job dọn dẹp), CORS, phục vụ frontend
  config.py            cấu hình (.env, tiền tố CALO_)
  db.py                bảng SQLAlchemy
  schemas.py           hợp đồng JSON (Pydantic)
  errors.py            mã lỗi chung
  deps.py              dependency: DB session, X-Device-Id
  routers/             scan.py (scan, recalc), meals.py, info.py (classes, benchmark, health)
  services/            inference.py (hàng chờ GPU), catalog.py, calc.py, image_check.py, cleanup.py
  export_corrections.py
  mock/scan.json       kết quả thật của examples/B_170.jpg, dùng cho chế độ mock
  tests/
```
