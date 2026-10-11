# Kết quả baseline CRNN trên IIIT5K

**Kết quả thực chạy**, lấy từ checkpoint, dự đoán và lịch sử huấn luyện lưu trong dự án. Đây là baseline đầu tiên cho đề tài; chưa có so sánh nhiều seed hoặc mô hình tiền huấn luyện.

## Cấu hình và nguồn số liệu

- Run: `runs/baseline_seed42`; seed trọng số: 42.
- Train: 1800 ảnh, validation: 200 ảnh; test: 3000 ảnh. Các từ giống nhau sau chuẩn hóa không chia qua train/validation.
- Hoàn thành 50 epoch; checkpoint chọn ở epoch **47** theo Word Accuracy validation và loss thấp hơn khi bằng điểm.
- Validation tại checkpoint: Word Accuracy **28.50%**, CER **34.81%**.
- PyTorch: 2.5.1+cu121; CNN/BiLSTM trên cuda, CTC loss trên cpu.
- Batch 16, Adam lr=0.001; BiLSTM hai tầng, hidden 256 mỗi hướng, 36 ký tự và blank, greedy không từ điển.
- Padding batch: `cnn_right_white_margin_24_v1`; giữ tỷ lệ ảnh và độ dài hợp lệ, thêm ngữ cảnh trắng cho biên CNN.
- Thời gian cộng từ lịch sử: 343.3 giây. Đây là thời gian quan sát trên máy hiện tại có thể có tác vụ khác, không phải benchmark tốc độ chuẩn.
- Nguồn: [config](../runs/baseline_seed42/config.json), [môi trường](../runs/baseline_seed42/environment.json), [history](../runs/baseline_seed42/history.csv), [metadata đánh giá](../runs/baseline_seed42/evaluation/evaluation_metadata.json).

## Đánh giá trên ảnh test gốc

Word Accuracy: **22.37%** (671/3000 từ). Wilson 95%: **20.91%–23.89%**.

CER micro: **44.91%**, tương ứng 6858 phép sửa trên 15269 ký tự nhãn thật. CER không phải tỷ lệ từ sai.

Cùng checkpoint đạt Word Accuracy **100.00%** và CER **0.00%** trên train khi đánh giá ở chế độ suy luận. Khoảng cách với test gợi ý mô hình học sát dữ liệu train nhưng tổng quát hóa còn hạn chế; khác biệt phân phối giữa các phần dữ liệu cũng có thể góp phần.

## Phân tích bổ sung về từ đã gặp

Phân tích thăm dò sau khi đọc kết quả baseline; không dùng để chọn checkpoint. Các ảnh vẫn là test gốc, phân nhóm bằng nhãn từ đã/ chưa xuất hiện trong train.

| Nhãn từ | n | Word Accuracy | CER |
|---|---:|---:|---:|
| seen_in_train | 1261 | 38.07% | 35.21% |
| unseen_in_train | 1739 | 10.98% | 49.99% |

Chênh lệch hai nhóm không tự chứng minh cơ chế ghi nhớ; độ dài và phân phối ảnh cũng khác nhau. Validation chia theo nhóm từ còn test giữ split gốc, vì vậy không coi độ khó hai phần là giống nhau.

## Theo độ dài nhãn

| Nhóm | n | Word Accuracy | Wilson 95% | CER micro |
|---|---:|---:|---|---:|
| 1-3 | 866 | 37.64% | 34.48%–40.92% | 43.75% |
| 4-6 | 1355 | 21.03% | 18.95%–23.28% | 43.17% |
| 7-9 | 623 | 8.83% | 6.85%–11.32% | 47.22% |
| >=10 | 156 | 3.21% | 1.38%–7.28% | 46.38% |

Đây là kết quả phân tầng quan sát của lần chạy này. Word Accuracy yêu cầu đúng toàn bộ từ, nên cần xem cùng CER và số mẫu trước khi giải thích tác động của độ dài.

## Theo chỉ số sắc nét

Ngưỡng được tính từ train: **195.8705 / 684.3447**, phân vị 1/3 và 2/3. Điểm là phương sai Laplacian ảnh xám cao 64 pixel, không dùng padding.

| Nhóm | n | Word Accuracy | Wilson 95% | CER micro |
|---|---:|---:|---|---:|
| low_sharpness | 917 | 26.06% | 23.33%–29.00% | 39.02% |
| medium_sharpness | 879 | 24.80% | 22.06%–27.76% | 40.08% |
| high_sharpness | 1204 | 17.77% | 15.72%–20.03% | 52.69% |

`low_sharpness` nghĩa chỉ số thấp, chưa phải nhãn mờ chuẩn. Font, nền, nhiễu và tương phản có thể ảnh hưởng điểm; bảng này không chứng minh quan hệ nhân quả.

Trong lần chạy này, accuracy nhóm điểm thấp là 26.06%, nhóm điểm cao là 17.77%. Nếu nhóm điểm cao kém hơn, phân tầng này không hỗ trợ giả thuyết đơn giản “mờ hơn thì lỗi nhiều hơn”. Cần kiểm tra nhãn mờ trực quan và xem riêng tác động blur có kiểm soát bên dưới.

## Mờ và biến dạng có kiểm soát

Cùng các ID ảnh test, cùng nhãn, checkpoint, tiền xử lý và decoder; mỗi biến đổi được đánh giá riêng. Delta accuracy = biến đổi − gốc, đơn vị điểm phần trăm. Delta CER cũng bằng biến đổi − gốc.

| Biến đổi | Mức | Accuracy | Delta accuracy (đpt) | CER | Delta CER (đpt) |
|---|---:|---:|---:|---:|---:|
| gaussian_blur | 1 | 21.27% | -1.10 | 47.14% | +2.23 |
| gaussian_blur | 2 | 14.77% | -7.60 | 58.94% | +14.02 |
| gaussian_blur | 3 | 8.13% | -14.23 | 71.98% | +27.07 |
| perspective | 1 | 17.37% | -5.00 | 52.39% | +7.47 |
| perspective | 2 | 9.63% | -12.73 | 65.19% | +20.28 |
| perspective | 3 | 5.07% | -17.30 | 78.09% | +33.17 |
| wave | 1 | 15.93% | -6.43 | 52.32% | +7.41 |
| wave | 2 | 11.77% | -10.60 | 57.08% | +12.17 |
| wave | 3 | 7.17% | -15.20 | 64.82% | +19.90 |

Khoảng tin cậy bootstrap theo cặp và số chuyển đúng→sai/sai→đúng nằm trong [báo cáo chi tiết](../runs/baseline_seed42/analysis/report.md). Biến đổi hình học còn có nội suy và thu nhỏ để giữ chữ trong canvas; không coi tác động này là đại diện đầy đủ cho mọi biến dạng tự nhiên.

## Biến dạng tự nhiên và giới hạn

Hiện có **0** ảnh test được gán nhãn biến dạng tự nhiên. Nếu số này bằng 0, chưa có kết quả phân tầng biến dạng tự nhiên; các ô trống không được tính là `none`. Cần gán nhãn theo [hướng dẫn](gan_nhan.md) để bổ sung phần này.

Baseline chỉ học từ đầu trên 1.800 ảnh IIIT5K. Bài CRNN gốc huấn luyện trên khoảng 8 triệu ảnh tổng hợp; hai kết quả khác giao thức và ngân sách dữ liệu, không thể dùng điểm baseline để kết luận tái lập bài gốc. Nguồn: [Shi, Bai và Yao](https://arxiv.org/pdf/1507.05717).

Đây là một lần chạy với một seed. Wilson mô tả bất định tỷ lệ theo giả định ảnh độc lập; không đo biến thiên giữa các lần khởi tạo. Giữ dữ liệu test cho đánh giá và công bố việc đã xem kết quả nếu tiếp tục điều chỉnh mô hình trên cùng benchmark.

Các bước tiếp theo cho đề tài: gán nhãn biến dạng tự nhiên, kiểm tra trực quan các nhóm mờ, thử tiền huấn luyện có nguồn rõ hoặc augmentation với lựa chọn bằng validation; báo riêng từng giao thức.

## Biểu đồ và ví dụ lỗi

![Đường học](../runs/baseline_seed42/figures/training_curves.png)

![Độ bền](../runs/baseline_seed42/figures/robustness_curves.png)

![Ảnh lỗi](../runs/baseline_seed42/figures/error_examples.png)

![Các tác động](../runs/baseline_seed42/figures/corruption_examples.png)

Bảng đầy đủ và dữ liệu từng ảnh: [analysis](../runs/baseline_seed42/analysis/report.md), [predictions](../runs/baseline_seed42/evaluation/predictions.csv).
