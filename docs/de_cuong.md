# Đề cương đề tài học sâu

**Tên đề tài:** Tìm hiểu kiến trúc CRNN kết hợp CNN–BiLSTM–CTC và ứng dụng nhận dạng từ trên IIIT5K; phân tích lỗi theo độ dài, độ mờ và biến dạng chữ.

**Trạng thái:** Đề cương và giao thức thực nghiệm. Baseline đã được triển khai; số liệu thật được lưu riêng trong [kết quả baseline](ket_qua_baseline.md). Các giả thuyết bên dưới là điểm xuất phát và phải đối chiếu với kết quả đo.

## 1. Bài toán và phạm vi

Đầu vào là ảnh đã cắt chứa một từ; đầu ra là chuỗi ký tự được đọc từ ảnh. Hệ thống nhận dạng tiếng Anh và chữ số, không phân biệt chữ hoa/chữ thường, với bảng chữ cái `0123456789abcdefghijklmnopqrstuvwxyz`. Mỗi ảnh có thể có số ký tự khác nhau. Đề tài không thực hiện tìm vị trí chữ trong ảnh toàn cảnh; muốn xây dựng OCR trên ảnh toàn cảnh cần bổ sung bộ phát hiện và cắt vùng chữ.

IIIT5K gồm 5.000 ảnh từ đã cắt, bao gồm chữ trong cảnh và ảnh được tạo sẵn bằng máy tính (born-digital). Phân chia gốc có 2.000 mẫu train và 3.000 mẫu test. Vì vậy tên đề tài nhấn mạnh ứng dụng trên IIIT5K, không giả định mọi mẫu đều là ảnh chụp cảnh tự nhiên. Nguồn: [Mishra, Alahari và Jawahar, BMVC 2012, mục 3.1](https://www.bmva-archive.org.uk/bmvc/2012/BMVC/paper127/paper127.pdf).

## 2. Mục tiêu

1. Giải thích cách CNN chuyển ảnh thành chuỗi đặc trưng, BiLSTM khai thác ngữ cảnh hai chiều và CTC học từ nhãn chuỗi khi chưa biết căn chỉnh từng ký tự.
2. Xây dựng mã PyTorch có thể huấn luyện và suy luận trên máy Windows có GPU NVIDIA.
3. Thiết lập baseline CRNN huấn luyện từ đầu trên phần train của IIIT5K, chọn checkpoint bằng validation và đánh giá bằng greedy decoding không dùng từ điển.
4. Đo Word Accuracy và micro Character Error Rate (CER); phân tích theo nhóm độ dài, chỉ số độ sắc nét và nhãn biến dạng tự nhiên.
5. Đánh giá độ bền bằng cách tạo ảnh mờ và biến dạng có kiểm soát trên cùng các ảnh test, giữ nguyên mô hình và nhãn.
6. Lưu cấu hình, seed, phân chia dữ liệu, checkpoint, dự đoán và biểu đồ để có thể tái lập và giải thích kết quả.

## 3. Câu hỏi nghiên cứu và giả thuyết

| Câu hỏi | Giả thuyết cần kiểm chứng | Cách kiểm chứng |
|---|---|---|
| Độ dài từ có liên quan thế nào đến lỗi? | Từ dài có thể giảm xác suất đúng toàn bộ; Word Accuracy giảm không nhất thiết đồng nghĩa CER tăng. | So sánh các nhóm 1–3, 4–6, 7–9 và từ 10 ký tự; báo cáo số mẫu và khoảng tin cậy. |
| Ảnh có chỉ số sắc nét thấp có khó nhận dạng hơn? | Nhóm có phương sai Laplacian thấp có thể có nhiều lỗi hơn. | Dùng ngưỡng xác định trên train, so sánh trên test; kiểm tra ví dụ ảnh để phát hiện ảnh hưởng của nền và độ tương phản. |
| Biến dạng tự nhiên ảnh hưởng thế nào? | Phối cảnh, chữ cong và kiểu chữ cách điệu có thể làm tăng lỗi của CRNN cơ sở. | Gán nhãn thủ công độc lập với dự đoán rồi phân tích theo nhóm. |
| Khi tăng mức tác động nhân tạo, độ chính xác thay đổi thế nào? | Mờ, phối cảnh và uốn sóng có thể làm tăng lỗi khi mức tác động tăng. | Đánh giá từng ảnh gốc và ảnh biến đổi theo cặp; vẽ đường độ bền. |

Những phát biểu này chưa phải kết luận. Lỗi giữa các nhóm trên dữ liệu gốc là mối liên hệ quan sát, có thể bị ảnh hưởng bởi độ dài, kích thước, nền ảnh hoặc kiểu chữ. Thử nghiệm tác động có kiểm soát giúp so sánh trên cùng ảnh, nhưng không đại diện đầy đủ cho mọi biến dạng trong đời thực.

## 4. Hướng triển khai

Mô hình triển khai dùng ảnh xám cao 32, CNN lấy chuỗi theo chiều ngang, BiLSTM hai tầng với hidden size 256 mỗi hướng và lớp tuyến tính 37 đầu ra gồm 36 ký tự cùng CTC blank. Checkpoint được chọn bằng Word Accuracy trên validation; CER dùng để bổ sung thông tin. Giao thức chi tiết nằm ở [thiết kế thực nghiệm](thiet_ke_thuc_nghiem.md), nền tảng toán học ở [cơ sở lý thuyết](co_so_ly_thuyet.md).

CRNN gốc đã huấn luyện trên khoảng 8 triệu ảnh tổng hợp rồi đánh giá các benchmark. Baseline của đề tài huấn luyện từ đầu trên dữ liệu nhỏ hơn nhiều, nên không coi là tái lập trực tiếp kết quả bài báo và không đặt trước một mức chính xác phải đạt. Nguồn: [Shi, Bai và Yao, mục 3.1](https://arxiv.org/pdf/1507.05717).

Nếu bổ sung trọng số đã huấn luyện, cần công bố nguồn tải, giấy phép, kiến trúc, bảng ký tự, dữ liệu tiền huấn luyện và mọi lần tinh chỉnh. Kết quả phải có dòng riêng, dùng cùng tập test và cùng cách chuẩn hóa nhãn/giải mã.

## 5. Các mốc thực hiện

| Mốc | Công việc | Điều kiện hoàn thành |
|---|---|---|
| 1. Nền tảng | Đọc CRNN, LSTM/BiLSTM, CTC; mô tả IIIT5K. | Giải thích được luồng tensor và ví dụ CTC có ký tự lặp. |
| 2. Dữ liệu | Kiểm tra dữ liệu, nhãn, bảng ký tự; tạo train/validation. | Manifest cố định, audit rõ số mẫu, không trộn test vào train. |
| 3. Baseline | Kiểm thử tính khả thi CTC và học thử một tập nhỏ; huấn luyện đầy đủ. | Loss hữu hạn, checkpoint và lịch sử học được lưu. |
| 4. Đánh giá | Chạy checkpoint đã chọn trên test. | Dự đoán từng ảnh, Word Accuracy, CER, khoảng tin cậy. |
| 5. Phân tích | Gán nhãn biến dạng, phân nhóm độ dài/mờ, chạy tác động nhân tạo. | Bảng nhóm, đường độ bền, ảnh lỗi minh họa có chú thích. |
| 6. Báo cáo | Viết kết quả, giới hạn và hướng cải tiến. | Mỗi con số truy được về tệp kết quả; tách kết quả đo khỏi giả thuyết. |

Thời lượng các mốc phụ thuộc GPU, khả năng hội tụ và việc gán nhãn; không suy thời gian huấn luyện hiện tại từ thời gian trong bài báo.

## 6. Sản phẩm bàn giao

- Mã chuẩn bị dữ liệu, mô hình, huấn luyện, suy luận, đánh giá và phân tích lỗi.
- Cấu hình thí nghiệm, manifest dữ liệu, nhật ký học và checkpoint tốt nhất.
- Dự đoán từng ảnh và báo cáo nhóm, biểu đồ cùng ảnh lỗi minh họa.
- Đề cương, cơ sở lý thuyết, giao thức thực nghiệm và [mẫu báo cáo](mau_bao_cao.md).

## 7. Tiêu chí đánh giá đề tài

Đề tài đạt yêu cầu khi mô hình được triển khai đúng, thực nghiệm có thể kiểm tra lại, không rò rỉ thông tin từ test để chọn mô hình, và phân tích lỗi có căn cứ. Độ chính xác thấp vẫn có giá trị nếu chỉ ra được nguyên nhân, giới hạn dữ liệu và bằng chứng rõ ràng; không thay thế số liệu thật bằng số liệu minh họa.

