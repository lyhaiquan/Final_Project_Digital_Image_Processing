# Hướng dẫn chạy thử backend + frontend (nhánh `feature/nhi`)

Nhánh này gộp **backend FastAPI + MySQL** và **frontend** (nhánh `fe`). Làm theo thứ tự dưới đây trên máy sạch là chạy được toàn bộ web app với mô hình thật. Thời gian lần đầu: khoảng 20–30 phút, chủ yếu là tải thư viện.

Lệnh viết cho **PowerShell trên Windows**.

---

## 0. Cần cài sẵn

| Phần mềm | Phiên bản | Kiểm tra |
|---|---|---|
| Python | 3.10 – 3.12 | `python --version` |
| Node.js | 20.19+ hoặc 22.12+ | `node --version` |
| Docker Desktop | bản mới, **đang bật** (Engine running) | `docker info --format "{{.ServerVersion}}"` |
| Git | bất kỳ | `git --version` |

Lần chạy đầu cần mạng để tải DINOv2 (khoảng 330 MB) và image MySQL (khoảng 200 MB).

---

## 1. Lấy code

```powershell
git clone https://github.com/lyhaiquan/Final_Project_Digital_Image_Processing.git
cd Final_Project_Digital_Image_Processing
git checkout feature/nhi
```

Nếu đã có repo thì chỉ cần: `git fetch origin` rồi `git checkout feature/nhi` và `git pull`.

## 2. Cài Python

```powershell
python -m venv .venv
.venv\Scripts\activate
# chọn MỘT trong hai dòng:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # máy có GPU NVIDIA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # máy không có GPU
pip install -r requirements.txt -r requirements-backend.txt
```

> PowerShell báo *"running scripts is disabled"* thì chạy một lần `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 3. Bật MySQL bằng Docker

```powershell
Copy-Item .env.example .env
docker compose up -d
docker compose logs mysql | Select-String "ready for connections"
```

Thấy dòng `ready for connections` (chờ khoảng 20–30 giây sau lần `up` đầu) là xong.

> **Máy đã cài MySQL sẵn ở cổng 3306** (lỗi `ports are not available ... 3306`): mở `.env`, đổi `MYSQL_PORT=3307` và cổng trong
> `CALO_DATABASE_URL` thành `localhost:3307`, rồi chạy lại `docker compose up -d`.

## 4. Chạy backend (terminal 1)

```powershell
.venv\Scripts\activate
uvicorn backend.app:app --port 8000
```

Chờ log hiện `Đã nạp model food_seg_v4_final+dino` (khoảng 10–20 giây; lần đầu lâu hơn vì tải DINOv2).
Mở <http://localhost:8000/api/health> → phải thấy `"status":"ok"`. Tài liệu API: <http://localhost:8000/docs>.

## 5. Chạy frontend (terminal 2)

```powershell
cd frontend
npm install
Set-Content .env "VITE_USE_MOCK=false`nVITE_API_BASE_URL=http://localhost:8000/api" -Encoding utf8
npm run dev
```

Mở <http://127.0.0.1:5173>. Giao diện **không còn** banner "mô phỏng" là đã nối backend thật.

---

## 6. Kịch bản test

Ảnh mẫu nằm trong thư mục `examples/`. Kết quả mô hình là cố định với cùng ảnh (NFR-B05); trên GPU có thể lệch vài gram.

| # | Thao tác | Kết quả đúng | Yêu cầu SRS |
|---|---|---|---|
| 1 | Trang **Quét** → chọn `examples/B_170.jpg` → **Phân tích ảnh** | Viền món hiện trên ảnh; **6 món, 464 g, 692 kcal**; 3 món viền cam "? Cần xác nhận" (su su, thịt kho, gà xào ớt chuông) | UC-F01, FR-F05–F08 |
| 2 | Bấm một lựa chọn trong thẻ hỏi "Vùng màu cam là món gì?" | Số cập nhật ngay; viền món vẫn giữ; thẻ chuyển sang món tiếp theo | UC-F02, FR-F09 |
| 3 | Bấm một dòng món → **Đổi món** sang món khác | Gram tính lại theo công thức của món mới; **các món khác giữ nguyên kcal** | UC-F03, FR-B12 |
| 4 | Dòng cơm → bấm suất **150 / 227 / 300 g** | Gram cơm đổi đúng suất | UC-F06, FR-B14 |
| 5 | **Xóa** một món → **Hoàn tác** | Món quay lại, tổng như cũ | UC-F05, FR-F12 |
| 6 | **+ Thêm món bị sót** → gõ `ca vien` | Tìm ra "cá viên"; món mới có nhãn thêm tay, gram mặc định, không có viền | UC-F04, FR-F13 |
| 7 | **Lưu vào nhật ký** (bật "Góp ảnh") → trang **Nhật ký** | Bữa hiện đúng giờ và tổng kcal; % so với 2.000 kcal; xóa được (có hỏi lại) | UC-F07–F08, FR-B15 |
| 8 | Quét `examples/B_212.jpg` | **6 món, 487 g, 524 kcal**; 1 món cần xác nhận (rau xào thập cẩm) | NFR-B05 |
| 9 | Chọn một file không phải ảnh (vd. đổi tên `.txt` thành `.jpg`) | Báo "Ảnh không hợp lệ" | FR-B01, bảng 2.6 |
| 10 | Trang **Bảng calo** và **Benchmark** | 45 món; Benchmark có 4 chỉ số, so sánh YOLO và DINOv2, AP từng món | FR-F15–F16 |

**Kiểm tra dữ liệu đã vào MySQL** sau bước 7:

```powershell
docker compose exec mysql mysql -ucalo -pcalo calo_khay -e "SELECT meal_id, eaten_at, total_kcal FROM meals; SELECT meal_id, cls, grams, kcal, source, confirmed FROM meal_items;"
```

`eaten_at` lưu theo giờ UTC (giờ Việt Nam trừ 7 tiếng), đúng như thiết kế.

**Test tự động** (không cần MySQL hay GPU):

```powershell
python -m pytest backend/tests -q      # backend: 25 passed
cd frontend; npm test                  # frontend: 7 pass
```

---

## 7. Những điểm cần lưu ý khi đánh giá

- **Tốc độ:** trên CPU khoảng 3 giây/ảnh (laptop thành viên test). Mục tiêu ≤ 1 giây của NFR-B01 tính cho GPU, chưa đo trên laptop GPU của nhóm.
- **Quyết định đã chốt** (ngưỡng cần xác nhận 0,75, bật DINOv2, tắt độ sâu mặc định, MySQL thay SQLite...): xem bảng ở mục 2 của
  [backend/README.md](../backend/README.md). Một số điểm khác SRS, cần cập nhật lại tài liệu đặc tả.
- **Trang Benchmark** hiện số của cấu hình mặc định (tắt độ sâu): 26,5 g/món, 111 kcal/khay. Bật độ sâu: thêm `CALO_DEPTH_ENABLED=true` vào `.env` (cần GPU, tải thêm khoảng 1,3 GB).
- **Ngưỡng ảnh mờ** (`CALO_BLUR_THRESHOLD=40`) đang đặt tạm, cần chỉnh theo ảnh chụp thật ở căng tin.

## 8. Gặp lỗi

| Lỗi | Cách xử lý |
|---|---|
| Backend: `Can't connect to MySQL server` | MySQL chưa sẵn sàng hoặc sai cổng: xem lại bước 3 và cổng trong `.env` |
| Backend: `Access denied for user 'calo'` | `docker compose down -v` rồi `docker compose up -d` (xóa dữ liệu MySQL cũ) |
| `/api/health` trả `loading` mãi | Đang tải DINOv2 lần đầu: chờ thêm; xem log ở terminal 1 |
| Frontend báo "Không thể kết nối máy chủ" | Backend chưa chạy ở cổng 8000, hoặc `frontend/.env` sai: sửa xong phải tắt `npm run dev` và chạy lại |
| Chỉ muốn xem giao diện, không chạy mô hình | Không tạo `frontend/.env` (mặc định chế độ mock) |

## 9. Dọn dẹp

```powershell
# Ctrl+C ở hai terminal, rồi:
docker compose down        # tắt MySQL, giữ dữ liệu
docker compose down -v     # tắt và xóa luôn dữ liệu
```
