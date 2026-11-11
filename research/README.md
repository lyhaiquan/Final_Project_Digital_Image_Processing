# Code nghiên cứu (để tham khảo)

Đây là code đã dùng để chuẩn bị dữ liệu, train và chấm điểm. **Bộ dữ liệu không có trong repo**, nên các script này
giữ nguyên đường dẫn gốc trên máy người train (`D:/food_dataset_v4`, `D:/food-labeling/...`) và sẽ không chạy được
nếu thiếu dữ liệu. Để *dùng* mô hình, chỉ cần `predict.py` ở thư mục gốc.

| Thư mục | Nội dung |
|---|---|
| `data_prep/` | Gom ảnh, chia train/val/test theo buổi chụp (`make_split.py`, `build_v*.py`), sửa/gộp nhãn v3 → v4 (`relabel.py`), bảng calo (`calories.py`) |
| `notebooks/` | Notebook train trên Kaggle (`train_v3`, `train_v4`, `train_v4_full` = model cuối), bản Colab, và 2 notebook tính đặc trưng độ sâu (`depth_check`, `depth_pred`) |
| `evaluation/` | Chấm ước lượng gram (`grams.py`), tìm đĩa (`plate.py`), bảng nhầm lẫn (`confusion.py`), ảnh lỗi (`show_errors.py`), đánh giá độ sâu (`depth_eval*.py`) |
| `improve/` | Thử nghiệm cải thiện sau khi train, không train lại YOLO: phân loại lại bằng DINOv2, gộp seed, hệ số gram, cờ xác nhận. Kết quả ở `improve/README.md` |

Kết quả các lần chạy nằm ở `docs/results/`; báo cáo đầy đủ ở `docs/benchmark_ieee.pdf`.
