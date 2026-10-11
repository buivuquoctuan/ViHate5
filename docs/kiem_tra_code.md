# Kiểm tra code và dọn dự án ngày 11/10/2026

Pipeline CNN–BiLSTM–CTC hoạt động theo thiết kế của đề tài. Kết luận này dựa trên kiểm thử và chạy thực tế dưới đây; không có nghĩa mô hình nhận dạng đúng mọi ảnh hoặc mọi kết luận nghiên cứu đã hoàn tất.

## Kiểm tra đã thực hiện

| Hạng mục | Kết quả |
|---|---|
| Toàn bộ kiểm thử CPU | **140 passed** |
| Phụ thuộc môi trường | `pip check`: không có phụ thuộc lỗi |
| Thiết bị | PyTorch 2.5.1+cu121 dùng được RTX 3050 Laptop |
| Huấn luyện → đánh giá → phân tích theo cặp trên GPU | Smoke test một epoch đạt; dữ liệu tổng hợp nhỏ, không dùng điểm số này trong báo cáo IIIT5K |
| Suy luận checkpoint cải tiến | Hai ảnh test mẫu được nhận dạng `private` và `parking` |
| Sáu run nghiên cứu | Hash checkpoint, quy tắc chọn epoch, ID/nhãn test và quy ước đánh giá khớp |
| Giao diện gán nhãn offline | Chín bước kiểm tra trình duyệt đạt, gồm chuyển trang, lưu và tải CSV/metadata |
| Script PowerShell | Cú pháp hợp lệ; run tồn tại bị từ chối trước khi thực hiện pipeline |

Các kiểm thử hiện có kiểm tra hình dạng tensor, độ dài hợp lệ/padding, CTC với ký tự lặp, tương thích checkpoint, train-only augmentation, warm start, scheduler/early stopping, công thức Accuracy/CER, split theo nhóm từ, provenance và import nhãn. Chúng không phải các phép thử thay thế đánh giá chất lượng nhận dạng trên dữ liệu thật.

## Những điểm đã sửa trong đợt rà soát

- `evaluate.py` và `prepare.py` kiểm tra đầu ra tồn tại trước khi đọc checkpoint hoặc quét ảnh. Tránh chạy tính toán rồi mới phát hiện tên thư mục bị trùng.
- `run_extended.py` từ chối stage dang dở trước khi ghi metadata nguồn, bảo toàn snapshot của lần chạy cũ.
- `run_baseline.ps1` tự đặt tên thư mục theo thời gian, từ chối run đã có artifact và giữ seed chia dữ liệu cố định 42 khi thay seed huấn luyện. Script cũng xuất thống kê phép sửa ký tự.
- Hướng dẫn dùng tên đầu ra mới; không yêu cầu tải/chuẩn bị dữ liệu lại trong workspace hiện tại.
- Lựa chọn `clear`/`none` của công cụ gán nhãn đã có option đầy đủ và được kiểm tra trực tiếp trong trình duyệt.
- Phân tích run cải tiến chính được bổ sung đủ chín điều kiện độ bền đã đánh giá, kèm hình và [báo cáo từ số liệu thật](bao_cao.md).

## Tệp đã dọn

Đã giải phóng **2,52 GiB**, gồm cache tải thư viện và profile trình duyệt thử, `.pytest_cache`, bytecode `__pycache__`, bốn run smoke, ảnh preview tổng hợp và HTML/log kiểm tra giao diện tạm. Đã xóa checkpoint không còn tương thích của lần thử trước sửa padding; config/history của lần đó vẫn giữ để đối chiếu nhật ký.

Danh sách đường dẫn và dung lượng từng mục: [cleanup_20261011.json](cleanup_20261011.json).

Giữ nguyên `.venv`, metadata cài đặt editable trong `src/crnn_iiit5k.egg-info`, dữ liệu gốc/manifests/tổng hợp, công cụ gán nhãn, sáu checkpoint chính, checkpoint học trước, run tái lập baseline, dự đoán/bảng/hình, mã nguồn, kiểm thử, tài liệu và notebook. Cache sẽ được tạo lại khi chạy các công cụ tương ứng. Bộ smoke có thể sinh lại bằng script, không cần lưu trọng số thử của từng lần kiểm tra.

## Kết quả và phần còn thiếu

Baseline trung bình ba seed đạt **23,53% Accuracy**, phiên bản cải tiến **61,06%**. Run chính `improved_seed42`, chọn bằng validation, đạt **62,10% Accuracy**, **18,65% CER**. Số liệu đến từ toàn bộ 3.000 ảnh test; xem [bảng so sánh](../runs/comparison/comparison.md).

Nhãn độ mờ và biến dạng tự nhiên do người gán hiện còn **0/3.000**. Vì vậy phần quan sát tự nhiên chưa thể kết luận theo nhóm nhãn thủ công; các thử nghiệm biến đổi nhân tạo đã có kết quả riêng. Đợt cải tiến thay nhiều yếu tố đồng thời, chưa có ablation để tách đóng góp riêng từng yếu tố.

Thực hiện theo [hướng dẫn chạy từng bước](huong_dan_chay.md). Bắt đầu bằng checkpoint có sẵn để nhận dạng ảnh; huấn luyện lại là một thao tác riêng vào thư mục mới.
