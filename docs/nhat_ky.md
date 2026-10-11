# Nhật ký thực hiện

Ngày bắt đầu: **07/10/2026** (giờ Việt Nam).

- Khởi tạo mã CRNN PyTorch, chuẩn bị IIIT5K, huấn luyện, đánh giá và phân tích lỗi.
- Viết đề cương, cơ sở lý thuyết, giao thức và mẫu báo cáo bằng tiếng Việt, dẫn tài liệu gốc.
- Phát hiện GPU RTX 3050 Laptop 4 GB, driver 566.07; tạo môi trường Python 3.12 riêng trong `.venv`.
- Đã cài PyTorch 2.5.1 CUDA 12.1 và các thư viện trong `.venv`; `pip check` đạt. Phiên bản thực tế nằm trong `requirements-lock.txt`.
- Đã tải archive IIIT5K chính thức và xác minh SHA256. Audit đủ 5.000 ảnh: 1.800 train, 200 validation, 3.000 test; không trùng byte ảnh giữa các phần, không trùng từ giữa train/validation; tất cả mẫu có căn chỉnh CTC khả thi.
- Kiểm tra ảnh đơn phát hiện CNN bị ảnh hưởng bởi biên padding khi đổi batch. Đã thêm 24 pixel ngữ cảnh trắng bên phải, loại phần này khỏi độ dài BiLSTM/CTC, kiểm tra toàn bộ CNN–BiLSTM cho dự đoán nhất quán. Checkpoint định dạng v2 lưu chính sách padding.
- Lần chạy khám phá trước sửa được dừng ở epoch 37, giữ tại `runs/exploratory_no_cnn_margin_seed42`; không dùng lần này để công bố kết quả test. Tập test chưa được đánh giá khi sửa lỗi kỹ thuật.
- 48 kiểm thử đạt. Kiểm thử tích hợp GPU và đối chiếu ảnh đơn/batch đạt tại `runs/smoke_final`; đây là ảnh tổng hợp để kiểm tra mã.
- Baseline chính thức tại `runs/baseline_seed42` hoàn thành 50 epoch, batch 16, seed 42. Checkpoint chọn bằng validation ở epoch 47: Word Accuracy validation 28,50%.
- Trên 3.000 ảnh test gốc: đúng 671 từ, Word Accuracy 22,37%; CER micro 44,91% (6.858 phép sửa / 15.269 ký tự). Cùng checkpoint đúng 1.800/1.800 từ train ở chế độ suy luận; khả năng tổng quát hóa còn hạn chế.
- Đánh giá độ bền dùng cùng checkpoint trên ba mức của Gaussian blur, perspective và wave. Bảng, khoảng tin cậy, biểu đồ và dự đoán nằm trong run chính thức; [kết quả baseline](ket_qua_baseline.md) tổng hợp số liệu thật.

Nhãn biến dạng tự nhiên còn cần quan sát/gán nhãn; các ô trống chưa được xem là `none`. Phân tích biến đổi nhân tạo không thay thế cho nhãn biến dạng tự nhiên. Các thử nghiệm cải tiến tiếp theo cần được báo riêng và công bố rằng kết quả baseline trên test đã được xem.

## Rà soát ngày 11/10/2026

- Đợt học trước và thí nghiệm nhiều seed đã hoàn tất. Ba seed baseline có Accuracy test trung bình 23,53%; ba seed cải tiến 61,06%. Run chính seed 42 được chọn bằng validation, đạt 62,10% Accuracy và 18,65% CER.
- Đối chiếu lại sáu checkpoint với metadata, toàn bộ ID/nhãn test và lịch sử chọn epoch; các kiểm tra đạt. Nhận dạng GPU hai ảnh mẫu cho kết quả `private` và `parking`.
- Đã sửa bảo vệ đầu ra để dừng sớm khi thư mục cũ tồn tại; bảo vệ snapshot nguồn của run dang dở. Script baseline tách seed học khỏi seed chia dữ liệu cố định 42 và tự chọn tên run mới.
- Các lựa chọn `clear`/`none` trong giao diện gán nhãn đã có đầy đủ option. Kiểm tra trình duyệt đạt chín bước, gồm lưu tiến độ, chuyển hai trang ảnh và tải CSV/metadata. Nhãn chuẩn vẫn trống.
- Bổ sung đủ chín so sánh độ bền vào phân tích run chính, tạo hình và [báo cáo đề tài](bao_cao.md) từ artifact thực tế.
- Dọn cache, bytecode và các run smoke cũ; checkpoint thử trước sửa padding được bỏ, giữ config/history cho nhật ký. Các kiểm tra smoke có thể tạo lại bằng `scripts/smoke_test.py`; số liệu của chúng không nằm trong báo cáo nghiên cứu.
- Hướng dẫn thao tác: [chạy code](huong_dan_chay.md). Kết quả rà soát và danh sách dọn: [kiểm tra code](kiem_tra_code.md).
