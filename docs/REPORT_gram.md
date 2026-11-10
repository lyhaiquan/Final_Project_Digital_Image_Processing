# Ước lượng gram từ mask: kết quả (10/11/2026)

Script: `grams.py` (đánh giá), `plate.py` (đo diện tích đĩa bằng YOLOE prompt "plate").
Dữ liệu: 585 cặp (ảnh, món) có cân, 112 ảnh, **chỉ 38 khay khác nhau** (mỗi khay chụp khoảng 3 ảnh).
- Fit trên train+val (28 khay). **Chọn cách bằng CV theo nhóm buổi chụp.**
- Test: 156 cặp, 10 khay. Số test được in cho mọi cách nhưng **không dùng để chọn cách**.

## Cách làm (cách "hybrid")
- Diện tích món = diện tích mask / diện tích đĩa. Đĩa được tìm bằng YOLOE, chọn đĩa chứa nhiều đồ ăn dự đoán nhất. Nếu đĩa bị khung ảnh cắt thì fit elip để bù.
- Món khác cơm: `log g = a_món + b·log diện_tích`, với b dùng chung cho mọi món. Có thêm hiệu chỉnh smearing (×1.08) để bỏ sai lệch đoán thấp khi cộng tổng khay.
- **Cơm: dùng khẩu phần trung bình.** Trong CV, diện tích cơm cho kết quả tệ hơn khẩu phần trung bình (MAE 74 g so với 62 g). Giả thuyết: cơm hay bị món khác đè lên, còn gram cơm thật được tính bằng phép trừ (cân cả đĩa trừ đĩa, trừ món khác) nên bị nhiễu. Đây mới là giả thuyết, chưa kiểm chứng.

## Kết quả test (mask do model dự đoán, chuẩn hoá theo đĩa)
Model: `models/food_seg_v4_final.pt`. Bản 45 lớp, train trên **train+val**, 150 epoch, dùng `last.pt` (10/11).

| | Đoán theo khẩu phần TB (mốc) | Diện tích (hybrid) |
|---|---|---|
| MAE mỗi món | 35.8 g | **26.6 g** |
| Sai số % trung vị mỗi món | 39.9% | **30.5%** |
| MAE các món trừ cơm | 27.3 g | **15.8 g** |
| MAE cơm | 67.9 g | 67.9 g (dùng TB) |
| MAE khi lấy TB các ảnh cùng khay | 34.7 g | **25.8 g** |
| Món model không nhận ra (miss) | 41 / 156 | 41 / 156 |

Với mask nhãn tay (giới hạn trên của cách 2D): MAE 24.3 g.

## Phát hiện
1. **Diện tích làm sai số các món khác cơm giảm khoảng một nửa** (27.3 → 15.8 g).
2. **Cơm không dự đoán được từ ảnh 2D.** Trên 35 khay, diện tích cơm còn tương quan âm với gram (Spearman −0.37). Cơm là món nặng nhất nên chiếm phần lớn sai số tổng của cả khay.
3. **Cùng một khay, diện tích món lệch nhau trung bình 25% giữa các ảnh.** Nguyên nhân là món đè lên nhau và góc chụp nghiêng. Chuẩn hoá theo đĩa chỉ giảm được từ 28% xuống 25%.
4. **Lỗi lớn nhất của cả hệ thống nằm ở bước nhận diện món.** 26% món đã cân trong test không được nhận ra hoặc bị nhầm sang lớp khác, chủ yếu là: cá viên, gà chua ngọt, gà xào ớt chuông, rau muống, đậu rán (xem `confusion.py`).
5. **Tổng gram cả khay:** chưa có cách nào thắng rõ ràng. Trong CV, hybrid có MAE thấp hơn (99 g so với 112 g), nhưng sai số % trung vị gần như bằng nhau (21.4% so với 20.6%).

## Giới hạn
- Chỉ có 38 khay, test có 10 khay, nên các con số ở trên chỉ nên xem là ước lượng thô.
- Đồ đựng trong hộp: test có 9 món, MAE khoảng 54 g (đồ trên đĩa khoảng 24 g). Chuẩn hoá theo đĩa không áp dụng được cho hộp (ảnh B_335 không có đĩa).
- Diện tích PRED trong tập fit (train+val) lấy từ chính các ảnh model segmentation đã train, nên hiệu chỉnh cho PRED gần như giống hiệu chỉnh cho GT.

## Thử nghiệm Depth-Anything-V2 cho cơm (10/11)
Script: `kaggle/depth_check.ipynb` (Kaggle, notebook `food-depth-check`) và `depth_eval.py`. Mask dùng nhãn tay.
- Độ cao món = disparity của Depth-Anything-V2 trừ mặt phẳng fit trên **mặt đĩa lộ ra** (mask YOLOE). Chia cho độ trải disparity trong vùng đĩa để khử thang đo.
- "Thể tích" V = tổng độ cao dương trên mask món / diện tích đĩa.
- **Cơm:** tương quan Spearman giữa gram và đặc trưng, tính trên 35 khay:
  - diện tích: **−0.25**
  - thể tích, model Small: +0.35
  - thể tích, model **Large: +0.54** (t = 3.7, p ≈ 0.001)
- Cách hybrid, cơm tính bằng thể tích Large thay cho khẩu phần TB. Các món khác giữ cách tính theo diện tích (thêm độ sâu không giúp: CV 14.2 g ở cả hai).

| | CV: cơm khẩu phần TB | CV: **cơm theo độ sâu** | Test: cơm khẩu phần TB | Test: **cơm theo độ sâu** |
|---|---|---|---|---|
| MAE mỗi món | 22.9 g | **22.4 g** | 24.3 g | **21.9 g** |
| MAE cơm | 61.6 g | **58.8 g** | 68.4 g | **52.5 g** |
| MAE tổng khay | 99 g | **89 g** | 93 g | **75 g** |
| Sai số % trung vị tổng khay | 22% | **18%** | 19% | **17%** |

Giới hạn: kết quả trên dùng mask nhãn tay, chưa chạy với mask do model dự đoán. Phần cải thiện trong CV nhỏ hơn trên test. Số gram cơm thật vốn nhiễu vì được tính bằng phép trừ.

### Chạy lại với mask do model dự đoán (`depth_eval_pred.py`, notebook Kaggle `food-depth-pred`)
- **Cơm:** tương quan Spearman trên 35 khay:
  - diện tích: −0.18
  - thể tích: **+0.47**

| | CV: cơm khẩu phần TB | CV: **cơm theo độ sâu** | Test: cơm khẩu phần TB | Test: **cơm theo độ sâu** |
|---|---|---|---|---|
| MAE mỗi món | 23.0 g | 23.1 g | 26.8 g | **23.8 g** |
| MAE cơm | **61.6 g** | 62.4 g | 68.4 g | **53.8 g** |
| MAE tổng khay | 101 g | **94 g** | 113 g | **83 g** |
| Sai số % trung vị tổng khay | 21.4% | **18.7%** | 29.9% | **20.9%** |

Kết luận:
- Với cơm tính riêng từng đĩa, CV cho thấy độ sâu ngang khẩu phần TB.
- Với **tổng cả khay** (con số app hiển thị), độ sâu tốt hơn ở cả CV lẫn test.
- Model không nhận ra 41/156 món trong test. Lỗi này nằm ở bước nhận diện, độ sâu không sửa được.
