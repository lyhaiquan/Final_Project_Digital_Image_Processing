# Calo Khay · Frontend

Giao diện tiếng Việt theo hướng “Quét AI” trong `docs/NGHIEP_VU.md`. Dùng JavaScript modules + Vite, không cần cài Python hoặc nạp model để chạy giao diện.

## Chạy local

Yêu cầu Node.js 20.19+ hoặc 22.12+.

```powershell
cd frontend
npm install
npm run dev
```

Mở địa chỉ Vite in trong terminal (mặc định `http://127.0.0.1:5173`).

```powershell
npm test
npm run build
npm run preview
```

Build xuất vào `frontend/dist/`, có thể phục vụ bằng FastAPI hoặc static hosting. Điều hướng hash không cần cấu hình fallback routing.

## Tính năng

- Tải / kéo thả / chụp ảnh, kiểm tra định dạng, tối đa 10 MB, cạnh ngắn ít nhất 320 px; thu nhỏ cạnh dài về tối đa 1600 px.
- Ảnh mẫu, trạng thái phân tích có hủy; SVG đa giác co theo ảnh và liên kết tới dòng món ăn.
- Tổng gram/kcal, tỷ lệ nhu cầu ngày, xác nhận món nghi ngờ, đổi / thêm / xóa món và hoàn tác xóa; chỉnh gram, chọn suất cơm.
- Lưu bữa ăn, xem theo ngày và xóa có xác nhận; bảng 45 món tìm không dấu và sắp xếp; benchmark từ dữ liệu của repo.
- Bản nháp trong sessionStorage, nhật ký mock trong localStorage. Ảnh upload chỉ giữ bằng object URL trong bộ nhớ, không lưu vào nhật ký; tải lại trang vẫn giữ số liệu bản nháp nhưng phải chọn lại ảnh nếu muốn quét lại. Ảnh mẫu có thể phục hồi.

## Nối API sau

Sao chép `.env.example` thành `.env` rồi đổi:

```dotenv
VITE_USE_MOCK=false
VITE_API_BASE_URL=http://localhost:8000/api
```

Khởi động lại Vite hoặc build lại. Chỉ `src/api.js` giao tiếp backend; UI dùng cùng interface ở cả hai chế độ. Biến `VITE_*` là cấu hình công khai trong trình duyệt, không đặt secret ở đây.

| Endpoint | Hợp đồng frontend sử dụng |
| --- | --- |
| `GET /classes` | Mảng `{cls, name, kcal_per_100g, default_g, suggestions, source, note}` hoặc `{classes: [...]}` |
| `POST /scan?depth=true/false` | Multipart `image`. JSON đúng README gốc, thêm `scan_id` |
| `POST /recalc` | `{scan_id, items}` → `{items, total}`; giữ `id`, `polygons`, `needs_confirm`, `source` và các metadata món |
| `POST /meals` | `{scan_id, items, eaten_at, share_corrections}` → `{meal_id}` |
| `GET /meals?date=YYYY-MM-DD` | `{meals: [{meal_id, eaten_at, items, total}], total}` |
| `DELETE /meals/{id}` | HTTP 204 |
| `GET /benchmark` | Cấu trúc như `public/mock/benchmark.json` |

`src/api.js` gửi `X-Device-Id` ngẫu nhiên cho API thật. Backend cần thống nhất định danh thiết bị, cho phép origin frontend cùng header này trong CORS; cấu trúc bọc JSON có thể điều chỉnh tại adapter khi chốt backend. `eaten_at` gửi ISO UTC; ngày nhật ký và giờ hiển thị theo múi giờ thiết bị. Request timeout sau 15 giây. Không tự chuyển về mock khi API lỗi.

Khi đổi tên món, frontend bỏ `grams` và gửi `source: renamed` cùng `area_ratio` để backend tính lại khối lượng theo hệ số lớp mới. Khi chỉnh gram hoặc chọn suất cơm, `grams` là giá trị ghi đè. API phải trả lại các metadata vùng để giữ tương tác ảnh.

Mock trả cùng một fixture cho mọi ảnh; banner luôn nêu rõ đây là mô phỏng. Chỉ vẽ polygon fixture lên ảnh mẫu tương ứng, không vẽ lên ảnh người dùng. Mock không thực hiện suy luận AI và không thu thập ảnh; tùy chọn góp ảnh chỉ mở khi dùng API thật. Toggle độ sâu gửi tham số thật khi nối backend, không giả lập cải thiện chất lượng trong mock.

## Cập nhật dữ liệu mẫu

```powershell
python scripts/sync-data.py
```

Script dùng thư viện chuẩn Python, đọc `models/calories.csv`, `models/gram_model.json`, bảng AP trong `docs/results/`; sao chép `examples/B_212.jpg`. Polygon mẫu vẽ tay phục vụ kiểm thử UI, không phải kết quả suy luận. Số liệu benchmark tổng hợp theo README hiện tại; khi cập nhật báo cáo, cập nhật metadata tương ứng trong script.

## Cấu trúc

- `src/main.js`: UI và trạng thái màn hình.
- `src/style.css`: token màu, bố cục desktop/mobile.
- `src/api.js`: adapter mock/HTTP, timeout, lưu nhật ký mock.
- `src/domain.js`: tính tổng, tính lại mock, tìm kiếm, mã lỗi, chuẩn hóa polygon.
- `src/image.js`: kiểm tra và thu nhỏ ảnh.
- `public/mock/`: fixtures, danh mục và benchmark.
- `tests/domain.test.js`: tổng kcal, sửa khẩu phần, đổi lớp, mã lỗi, tọa độ và escaping.

Font Be Vietnam Pro tải từ Google Fonts; nếu offline, dùng Arial hệ thống. Không cần font mạng để sử dụng ứng dụng.
