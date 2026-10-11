# Gán nhãn độ mờ và biến dạng tự nhiên

`crnn.prepare` tạo `data/manifests/test_annotations.csv` với các cột `id,blur_manual,distortion,notes`. Mỗi ID liên kết một ảnh trong `test.csv`. Các ô trống có nghĩa **chưa gán nhãn**, không có nghĩa ảnh rõ hoặc chữ không biến dạng. Tệp gán nhãn đã tồn tại được giữ khi tạo lại manifest.

Xem ảnh gốc ở kích thước đủ lớn, gán nhãn trước khi xem dự đoán và giữ quy tắc nhất quán. Nên thử giao thức trên một phần train/validation rồi khóa tiêu chí. Nếu có hai người, gán nhãn độc lập một phần mẫu và báo mức đồng thuận; không tự đặt số người hoặc mức đồng thuận trong báo cáo.

| `distortion` | Tiêu chí nhìn ảnh |
|---|---|
| `none` | Chữ nằm gần đường thẳng, không thấy phối cảnh/uốn cong/kiểu cách đáng kể |
| `perspective` | Chữ hoặc đường nền bị nghiêng hội tụ, thay đổi tỷ lệ do góc chụp |
| `curved` | Đường nền chữ cong hoặc các ký tự bố trí trên cung |
| `stylized` | Font trang trí, nét cách điệu khiến hình dạng ký tự khác thường |
| `mixed` | Có từ hai loại trở lên, ghi rõ trong `notes` |

Font đậm/nghiêng thông thường không tự động là `stylized`. Ảnh có nhiều khoảng trắng hoặc tỷ lệ rộng/cao lớn không tự động là `perspective` hoặc `curved`. `none` chỉ điền sau khi đã xem ảnh.

`blur_manual` có thể dùng `clear`, `moderate`, `severe`: nét dễ phân biệt; nét mềm nhưng phần lớn còn đọc được; nét nhập nhòe làm khó phân biệt ký tự. Đây là đánh giá chủ quan cần ghi tiêu chí. Công cụ hiện phân nhóm độ mờ tự động bằng chỉ số Laplacian; cột này dùng để đối chiếu và mở rộng phân tích.

Nếu chỉ gán một phần test, báo tổng ảnh được gán và cách chọn mẫu. Không chỉ gán ảnh mô hình dự đoán sai rồi dùng tỷ lệ sai trong phần ấy làm ước lượng cho toàn bộ test. Nếu gán sau khi đã xem kết quả, công bố nguy cơ thiên lệch và tính chất khám phá.

Thử nghiệm Gaussian blur/perspective/wave do mã tạo ra được báo riêng theo mức biến đổi; không điền các nhãn nhân tạo vào bảng ảnh gốc để thay cho việc quan sát biến dạng tự nhiên.
