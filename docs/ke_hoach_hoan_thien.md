# Kế hoạch hoàn thiện thực nghiệm

Được viết trước các lần đánh giá test bổ sung, ngày **07/10/2026**. Baseline seed 42 đã có Word Accuracy test 22,37%; do kết quả này đã được xem, các thí nghiệm tiếp theo là thực nghiệm cải tiến có tính khám phá. Không gọi chúng là đánh giá trên benchmark hoàn toàn chưa được quan sát.

## Giao thức cố định

- Giữ nguyên manifests IIIT5K: 1.800 train, 200 validation, 3.000 test; không thay đổi bảng chữ cái, greedy decoder, xử lý ảnh hoặc chỉ số.
- Lặp baseline từ đầu với seed 7 và 123, thêm vào baseline seed 42 đang có. So ba lần khởi tạo trên cùng split; không chia lại dữ liệu.
- Tạo 20.000 ảnh train tổng hợp và 1.000 ảnh validation tổng hợp. Nguồn chữ gồm nhãn IIIT5K train và chuỗi chữ/số sinh ngẫu nhiên. Không đọc ảnh/nhãn IIIT5K validation/test hoặc từ điển đi kèm benchmark để tạo dữ liệu này.
- Validation tổng hợp dùng nhãn ngẫu nhiên chưa xuất hiện trong nguồn chữ train tổng hợp và các font Verdana/Tahoma được giữ riêng. Lưu hash font và nguồn chữ. Đây là bộ tự sinh nhỏ, không phải Synth90k chính thức.
- Huấn luyện CRNN trên bộ tổng hợp theo số epoch đã lưu trong config. Checkpoint được chọn bằng validation tổng hợp; có thể dùng validation IIIT5K để kiểm tra khả năng chuyển miền trước khi khóa công thức tinh chỉnh.
- Công thức cải tiến dự kiến: khởi tạo bằng cùng checkpoint tổng hợp, augmentation nhẹ chỉ trên train, dropout 0,2, AdamW với weight decay 0,0001, learning rate ban đầu 0,0003, ReduceLROnPlateau dựa trên validation loss và dừng sớm khi chỉ số chọn checkpoint không cải thiện. Mọi thay đổi công thức trước test phải được ghi lại cùng lý do dựa trên train/validation.
- Sau khi khóa công thức, chạy fine-tune ba seed 42, 7, 123. Không chọn giữa ba seed bằng test; báo cả ba, trung bình và độ lệch chuẩn mô tả.
- Đánh giá chín điều kiện mờ/biến dạng cho run cải tiến được chọn bằng validation. So với baseline trên cùng ID và nhãn, cùng decoder và tiền xử lý.

## Phân tích và báo cáo

Word Accuracy, micro CER, khoảng Wilson theo ảnh và biến thiên giữa seed được báo riêng. Với ba seed, độ lệch chuẩn chỉ mô tả ba lần chạy, không đủ cho kết luận thống kê mạnh về mọi khởi tạo. So sánh phiên bản thay nhiều yếu tố đồng thời không chứng minh yếu tố riêng nào tạo ra cải thiện.

Phân tích độ dài và Laplacian tiếp tục dùng ngưỡng train đã cố định. Bổ sung công cụ gán nhãn ảnh gốc, phân nhóm độ mờ thủ công `clear/moderate/severe`, biến dạng `none/perspective/curved/stylized/mixed`, tỷ lệ được gán và nguồn người gán. Ô trống vẫn là chưa gán. Nhãn đề xuất bởi AI nếu có phải được lưu riêng, công bố nguồn và chờ xác nhận; không ghi là nhãn chuẩn của IIIT5K.

Báo cáo sẽ giữ kết quả baseline thấp, trình bày đầy đủ các lần chạy mới, giới hạn việc dùng dữ liệu tổng hợp và phần gán nhãn còn chờ kiểm tra. Không đặt một mức accuracy phải đạt rồi điều chỉnh hoặc bỏ kết quả không đạt.

## Cơ sở lựa chọn

CRNN gốc học trên dữ liệu chữ tổng hợp quy mô lớn. Việc bổ sung dữ liệu tự sinh trong đề tài thử giảm sự phụ thuộc vào 1.800 ảnh thật; khả năng cải thiện phải được kiểm chứng bằng thí nghiệm. Nguồn: [CRNN](https://arxiv.org/pdf/1507.05717), [Synthetic Data and Artificial Neural Networks for Natural Scene Text Recognition](https://arxiv.org/abs/1406.2227).
