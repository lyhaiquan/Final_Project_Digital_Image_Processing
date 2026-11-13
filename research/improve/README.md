# Thử nghiệm cải thiện sau khi train (không train lại YOLO)

Cách làm: mọi lựa chọn (mô hình nền, ngưỡng, tham số) được quyết định trên **val** với 3 model chỉ-train
(`out_v4b/final_s{0,1,2}_best.pt`, tham khảo = polygon train). Tập **test** chỉ được chấm sau khi đã chốt cấu hình,
với 3 model full (`full_s{0,1,2}_last.pt`, tham khảo = polygon train + val). Model đang dùng trong repo là `full1`.

Cách chấm nhận diện giống `research/evaluation/confusion.py`: mỗi polygon nhãn tay lấy lớp dự đoán phủ nhiều nhất
(≥ 30% diện tích polygon), không có thì tính là bỏ sót. "Cặp" = (ảnh, món); món "ma" = món báo ra nhưng không có trong nhãn của ảnh.

## Kết quả

### 1. Phân loại lại bằng DINOv2 + hạ ngưỡng (ĐÃ ĐƯA VÀO pipeline)

Giữ mask của YOLO, lấy cả mask conf ≥ 0.1; mỗi vùng được phân loại lại bằng đặc trưng DINOv2-base (trung bình token patch
trong mask → hồi quy logistic), cộng log-xác suất với YOLO (trọng số 0.8 / 0.2); khử mask trùng (IoU ≥ 0.7);
mask conf < 0.25 chỉ giữ nếu xác suất kết hợp ≥ 0.5.

| Tập | Cách | Vùng đúng | Nhầm | Sót | F1 cặp | Món ma |
|---|---|---|---|---|---|---|
| val (411 vùng), TB 3 seed | YOLO | 227.7 | 93.7 | 89.7 | 70.4 | 68.7 |
| | + DINOv2 chỉ đổi nhãn | 246.7 | 74.7 | 89.7 | 76.5 | 49.0 |
| | + DINOv2 + hạ ngưỡng | 271.7 | 79.7 | 59.7 | 78.9 | 56.7 |
| test (382 vùng), TB 3 seed | YOLO | 260.7 | 77.7 | 43.7 | 77.3 | 44.0 |
| | + DINOv2 chỉ đổi nhãn | 271.3 | 67.0 | 43.7 | 82.9 | 28.7 |
| | + DINOv2 + hạ ngưỡng | 279.0 | 71.0 | 32.0 | 83.5 | 35.7 |
| test, model đang dùng | YOLO → mới | 266 → 279 | 79 → 74 | 37 → 29 | 79.5 → 84.5 | 40 → 33 |
| test, model đang dùng, chạy pipeline thật | YOLO → mới | 261 → 273 | 80 → 76 | 41 → 33 | 79.5 → 84.5 | 40 → 33 |

Từng seed trên test (vùng đúng): 253 → 281, 266 → 279, 263 → 277. Pipeline thật ra ít hơn vài vùng vì mask được chuyển thành
đa giác rút gọn trước khi chấm.

Tác động xuống gram và kcal, chạy bằng pipeline thật (`eval_pipeline.py`, 30 ảnh test có cân):

| Chỉ số | YOLO | + DINOv2 | YOLO, cơm theo độ sâu | + DINOv2, cơm theo độ sâu |
|---|---|---|---|---|
| Cặp có cân bị sót (trên 156) | 41 | 29 | 41 | 29 |
| MAE gram mỗi món | 26.8 | 26.5 | 23.8 | 24.5 |
| MAE gram tổng khay | 91.4 | 88.9 | 72.0 | 73.2 |
| MAE kcal tổng khay | 122.2 | 111.2 | 108.5 | 98.5 |
| Món sai không được gắn cờ | 27 | 14 | 27 | 14 |
| Tỉ lệ món bị gắn cờ | 13.7% | 25.2% | 13.7% | 25.2% |

Tổng khay ở đây cộng mọi món chương trình báo ra (kể cả món ma); món có nhãn nhưng không được cân (cơm của 2 khay test) thì bỏ qua.
Bảng V của báo cáo chỉ cộng các món có cân, nên cùng model chỉ-YOLO ra 113 g / 83 g; Bảng VI của báo cáo dùng cách tính ở đây.

Chi phí: thêm khoảng 0,2 s mỗi ảnh trên GPU, khoảng 2,5 s trên CPU; tải DINOv2-base 330 MB một lần.

Các lựa chọn đã thử trên val: DINOv2 small kém hơn (258 vùng đúng); DINOv2 large và SigLIP2 base ngang DINOv2 base;
ảnh crop 224 px ngang 336 px; token CLS của cả crop kém hẳn (≈ 40% khi mask hoàn hảo, so với 77% của đặc trưng trong mask).
DINOv3 không thử được vì model bị khóa trên Hugging Face.

### 2. Gộp 3 seed, TTA (KHÔNG đưa vào)

- TTA (lật ngang, thêm ảnh 800 px): không cải thiện trên val (226–229 so với 227.7 vùng đúng).
- Gộp 3 seed, nhãn YOLO: val 246 so với 227.7; test 267 so với 260.7.
- Gộp 3 seed + DINOv2: test 284 vùng đúng, F1 85.4 (một model + DINOv2: 279, 83.5). Nhận diện nhỉnh hơn một chút, nhưng sai số
  gram và kcal tổng khay không tốt hơn, mà chi phí YOLO gấp 3. Không đáng.

### 3. Co hệ số gram của từng món (ridge) (KHÔNG đưa vào)

`log g = mu + a_món + b·log A`, phạt `lam·a_món²`, thay cho quy tắc "món dưới 3 khay dùng hệ số chung".

| | CV: MAE món | CV: MAE tổng món khác cơm / ảnh | Test: MAE món | Test: MAE tổng / ảnh |
|---|---|---|---|---|
| Hiện tại | 14.2 | 46.7 | 15.4 | 25.5 |
| Ridge lam = 3 | 13.0 | 43.4 | 11.8 | 25.8 |

Sai số từng món giảm, nhưng mức giảm trên CV chỉ khoảng 1,5 sai số chuẩn, và khi ghép vào pipeline thì kcal tổng khay trên test
lại tăng (111 → 120). Bằng chứng chưa đủ để đổi.

### 4. Độ sâu cho món khác cơm (KHÔNG đưa vào)

Số liệu đã có từ `grams_out/depth_eval.csv`: thêm thể tích vào công thức không giảm sai số món khác cơm trên CV (14.2 → 14.2 g).

### 5. "±30% mỗi món" có đúng không

| Nhóm | ±30% phủ (CV / test) | ±50% phủ (CV / test) |
|---|---|---|
| Món khác cơm | 60% / 44% | 77% / 70% |
| Cơm, khẩu phần TB | 54% / 50% | 96% / 88% |
| Cơm, độ sâu | 63% / 79% | 86% / 96% |

`ERROR_PCT` trong pipeline vẫn để 30; muốn con số hiển thị đúng với khoảng 3/4 số món thì đổi thành 50.

### 6. Cờ "cần xác nhận"

Điểm dùng để gắn cờ = xác suất kết hợp DINOv2 + YOLO, ngưỡng 0.75 (chọn trên val để gắn cờ khoảng 25% số món).
Khả năng phân biệt món sai gần như không đổi (AUC test 0.75 → 0.77); cái được chủ yếu là số món sai ít đi.

### 7. Gộp các món nhìn gần giống nhau (KHÔNG đưa vào)

Nhóm chọn trước khi nhìn test, theo quy tắc trên val (`conf_val.py`): nhầm ≥ 3 lần mỗi seed và kcal/100 g chênh ≤ 20%.
- A: lạp xưởng + xúc xích, gà rán + gà sốt chua ngọt, cá viên + trứng cút, su su + rau xào thập cẩm.
- B: A + thịt kho + đậu sốt thịt (cặp nhầm nhiều nhất, nhưng kcal chênh 55%).

Cách làm (`merge_cls.py`, không train lại): cộng xác suất DINOv2 và YOLO trong nhóm rồi kết hợp như cũ, ngưỡng giữ nguyên;
fit lại hệ số gram theo nhóm; kcal/100 g của nhóm = trung bình theo gram của các thành viên. Sự thật vẫn tính theo kcal của từng món.
Để so công bằng, pipeline không gộp cũng được chấm ở cùng mức nhóm.

| Test, TB 3 seed | Vùng đúng (mức nhóm) | Món ma (mức nhóm) | Lỗi lọt không cờ (mức nhóm) | MAE kcal khay |
|---|---|---|---|---|
| Không gộp | 300.0 | 24.7 | 7.3 | 115.6 |
| Gộp A | 300.3 | 24.3 | 7.3 | 117.4 |
| Gộp B | 300.7 | 23.3 | 7.3 | 116.3 |

Tỉ lệ đúng tăng từ 73% lên 79% chỉ vì đổi cách chấm: bản không gộp, chấm theo nhóm, cũng ra đúng mức đó.
Sai số kcal tổng khay không giảm, mà còn tăng nhẹ ở cả 3 seed với phương án A (120.1 → 123.7, 111.2 → 111.3, 115.6 → 117.1).
Lý do: những cặp hay nhầm có kcal gần nhau nên nhầm giữa chúng vốn tốn ít calo; còn cặp kcal xa (B) thì gộp lại phải dùng kcal trung bình, sai thêm.

## Chạy lại

Cần bộ dữ liệu gốc và các model seed (đường dẫn `D:/...` trong `common.py`):

```
python cache_pred.py            # cache dự đoán 6 model
python embed.py dinov2b224      # đặc trưng DINOv2
python postproc.py dinov2b224   # dò tham số trên val
python merge.py; python embed.py dinov2b224 trE:val fullE:test
python test_rec.py              # nhận diện trên test
python downstream.py            # gram, kcal trên test (từ cache)
python eval_pipeline.py [--depth]   # pipeline thật trong repo
python gram2.py; python gram3.py; python gram4.py; python flag.py; python interval.py
python conf_val.py; python merge_cls.py
```
