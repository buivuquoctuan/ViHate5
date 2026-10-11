# Mẫu báo cáo đề tài CRNN trên IIIT5K

> Đây là mẫu để điền từ kết quả thực chạy. Mọi ô `⟨điền⟩` hoặc `—` là chưa có số liệu, không phải 0. Không dùng số liệu giả để hoàn thiện bảng. Nếu chưa chạy đủ một thí nghiệm, ghi “chưa thực hiện”.

## Thông tin chung

**Tên đề tài:** Tìm hiểu kiến trúc CRNN kết hợp CNN–BiLSTM–CTC và ứng dụng nhận dạng từ trên IIIT5K; phân tích lỗi theo độ dài, độ mờ và biến dạng chữ.

**Sinh viên/nhóm:** ⟨điền⟩  
**Giảng viên:** ⟨điền⟩  
**Mã lần chạy/config/checkpoint:** ⟨điền⟩  
**Ngày thực nghiệm:** ⟨điền theo giờ Việt Nam⟩

## Tóm tắt

Đề tài xây dựng mô hình CRNN gồm CNN, BiLSTM và CTC để nhận dạng ảnh đã cắt chứa một từ trên IIIT5K. Mô hình ⟨huấn luyện từ đầu/khởi tạo từ checkpoint có nguồn cụ thể⟩, dùng ⟨số mẫu train⟩ mẫu train và ⟨số mẫu validation⟩ mẫu validation, đánh giá không từ điển trên ⟨số mẫu test thực tế⟩ mẫu. Kết quả Word Accuracy là ⟨điền⟩, micro CER là ⟨điền⟩. Phân tích theo độ dài, chỉ số sắc nét và nhãn biến dạng cho thấy ⟨chỉ điền nhận xét có số liệu hỗ trợ⟩. Thử nghiệm tác động theo cặp cho thấy ⟨điền⟩. Giới hạn chính là ⟨điền⟩.

Nếu chưa có kết quả, thay đoạn số liệu bằng: “Hiện đã xây dựng cơ sở lý thuyết, mã và giao thức; kết quả huấn luyện và đánh giá sẽ được bổ sung sau thực nghiệm.”

## Chương 1. Giới thiệu

### 1.1. Bài toán

Trình bày ứng dụng đọc tên cửa hàng, biển hiệu hoặc nhãn sản phẩm. Xác định rõ đầu vào là vùng từ đã cắt và đầu ra là chuỗi tiếng Anh/chữ số không phân biệt hoa/thường. Mô hình trong đề tài không phát hiện vị trí chữ ở ảnh toàn cảnh.

### 1.2. Mục tiêu và câu hỏi nghiên cứu

Nêu ba câu hỏi chính: độ dài từ, mức sắc nét/mờ và biến dạng ảnh hưởng thế nào đến lỗi? Phân biệt mối liên hệ quan sát trên ảnh gốc với tác động được tạo nhân tạo trên cùng ảnh. Gắn mục tiêu với chỉ số và bảng sẽ báo.

### 1.3. Phạm vi và đóng góp

Đóng góp của đề tài là triển khai baseline có thể kiểm tra, xây dựng giao thức phân tích lỗi và giải thích kết quả. Không tuyên bố kiến trúc CRNN là kiến trúc mới. Không khẳng định đã tái lập độ chính xác bài báo nếu dữ liệu huấn luyện khác.

## Chương 2. Cơ sở lý thuyết

Phát triển nội dung từ [cơ sở lý thuyết](co_so_ly_thuyet.md): CNN/receptive field, chuyển bản đồ đặc trưng thành chuỗi, LSTM/BiLSTM, CTC và greedy decoding. Cần có sơ đồ mô hình, luồng kích thước tensor, công thức CTC và ví dụ `BOOK` có blank tách hai `O`.

Trích nguồn sơ cấp bằng tài liệu trong [BibTeX](tai_lieu_tham_khao.bib). Công bố tầng cuối `(2,1)` của mã khác cấu hình gốc; trình bày \(T_i=\lfloor W_i/4\rfloor\) và điều kiện \(T_i\ge L_i+r_i\).

## Chương 3. Dữ liệu và phương pháp

### 3.1. IIIT5K và phân chia dữ liệu

| Nội dung | Giá trị thực tế |
|---|---|
| Nguồn tải, phiên bản/hash | ⟨điền⟩ |
| Train gốc | 2.000 |
| Test gốc | 3.000 |
| Train dùng để học | ⟨điền từ manifest⟩ |
| Validation | ⟨điền từ manifest⟩ |
| Test thực sự đánh giá | ⟨điền⟩ |
| Seed chia dữ liệu | 42 hoặc ⟨giải thích thay đổi⟩ |
| Quy tắc nhóm validation | Các ảnh cùng nhãn chuẩn hóa ở cùng phần |
| Mẫu lỗi/loại và lý do | ⟨điền; nếu không có ghi 0⟩ |
| Số nhãn trùng train–validation | ⟨điền⟩ |
| Số nhãn trùng train gốc–test gốc | ⟨điền⟩ |

Mô tả IIIT5K chứa cả scene text và born-digital; không gọi toàn bộ là ảnh cảnh tự nhiên. Ghi cách chuẩn hóa hoa/thường và xử lý ký tự ngoài bảng.

### 3.2. Tiền xử lý và mô hình

Mô tả fit ảnh trong cao 32/rộng tối đa 320, giữ tỷ lệ, căn giữa dọc và padding phải. Ghi số ảnh có chiều cao chữ thực tế nhỏ hơn 32 do giới hạn chiều rộng. Mô tả chiều rộng tối thiểu 64, bội số 4, độ dài hợp lệ và packed BiLSTM.

| Cấu hình | Giá trị thực chạy |
|---|---|
| Bảng ký tự/blank | ⟨điền⟩ |
| Kênh CNN và tầng cuối | ⟨điền⟩ |
| BiLSTM | ⟨điền⟩ |
| Số tham số mô hình | ⟨điền từ chương trình⟩ |
| Optimizer | ⟨điền⟩ |
| Learning rate/scheduler | ⟨điền⟩ |
| Batch size | ⟨điền⟩ |
| Số epoch thực chạy | ⟨điền⟩ |
| Augmentation/dropout | ⟨điền⟩ |
| Precision, gradient clipping | ⟨điền⟩ |
| Thiết bị CTC loss | CPU (`ctc_loss_device=cpu`) hoặc ⟨công bố thay đổi⟩ |
| Seed trọng số | ⟨điền⟩ |
| Tiêu chí chọn checkpoint | Word Accuracy validation cao hơn; nếu hòa chọn CTC loss validation thấp hơn hoặc ⟨công bố thay đổi⟩ |
| Epoch checkpoint được chọn | ⟨điền⟩ |
| Decoder | Greedy không từ điển hoặc ⟨công bố thay đổi⟩ |

### 3.3. Môi trường

| Thành phần | Giá trị |
|---|---|
| Hệ điều hành | ⟨điền⟩ |
| Python/PyTorch | ⟨điền⟩ |
| GPU/VRAM | ⟨điền⟩ |
| Driver, CUDA của PyTorch | ⟨điền⟩ |
| Thời gian huấn luyện thực đo | ⟨điền⟩ |
| Tệp cấu hình và checkpoint | ⟨đường dẫn/hash⟩ |

### 3.4. Chỉ số và giao thức phân tích

Nêu công thức Word Accuracy, micro CER, Wilson 95%. Nêu nhóm độ dài, ngưỡng mờ lấy từ train và quy trình gán nhãn biến dạng. Liên hệ với [giao thức thực nghiệm](thiet_ke_thuc_nghiem.md). Công bố tham số tác động thực sự dùng; không chỉ ghi “mức 1, 2, 3”.

## Chương 4. Kết quả và phân tích

### 4.1. Quá trình huấn luyện

Chèn đường train/validation CTC loss và validation Word Accuracy/CER. Giải thích checkpoint đã chọn, tín hiệu quá khớp nếu có và sự khác biệt giữa học thử tập nhỏ với thực nghiệm đầy đủ. Không dùng train accuracy làm kết quả test.

### 4.2. Kết quả tổng thể

| Mô hình/giao thức | Nguồn khởi tạo | Số mẫu test | Word Accuracy (%) | Wilson 95% (%) | Micro CER (%) |
|---|---|---:|---:|---|---:|
| CRNN baseline | Từ đầu | — | — | — | — |
| Thí nghiệm bổ sung, nếu đã chạy | ⟨điền⟩ | — | — | — | — |

Nếu dẫn kết quả bài báo để tham khảo, tạo bảng riêng gồm dữ liệu huấn luyện, decoder, từ điển và quy tắc chọn mẫu. Không đặt kết quả dùng lexicon hoặc tiền huấn luyện hàng triệu ảnh thành so sánh ngang bằng với baseline nhỏ.

### 4.3. Lỗi theo độ dài

| Độ dài nhãn | Số mẫu | Số đúng | Word Accuracy (%) | Wilson 95% (%) | Micro CER (%) |
|---|---:|---:|---:|---|---:|
| 1–3 | — | — | — | — | — |
| 4–6 | — | — | — | — | — |
| 7–9 | — | — | — | — | — |
| Từ 10 | — | — | — | — | — |

**Nhận xét:** ⟨so sánh dựa trên số liệu, nhóm ít mẫu, Word Accuracy và CER có cùng xu hướng không; kiểm tra giới hạn chiều rộng và chữ lặp⟩.

### 4.4. Lỗi theo chỉ số sắc nét

Ngưỡng train: \(a=\) ⟨điền⟩, \(b=\) ⟨điền⟩; resize về cao 64 giữ tỷ lệ, không padding. Nêu thang pixel và toán tử Laplacian.

| Nhóm chỉ số sắc nét | Số mẫu | Word Accuracy (%) | Wilson 95% (%) | Micro CER (%) |
|---|---:|---:|---|---:|
| Thấp | — | — | — | — |
| Trung bình | — | — | — | — |
| Cao | — | — | — | — |

**Nhận xét:** ⟨mối liên hệ quan sát; ví dụ không khớp cảm nhận mờ; ảnh hưởng nền/độ tương phản⟩. Không gọi phương sai Laplacian là nhãn độ mờ chuẩn của IIIT5K.

### 4.5. Lỗi theo nhãn biến dạng tự nhiên

Số ảnh đã gán: ⟨điền⟩/⟨tổng test⟩. Số người gán và mức bất đồng: ⟨điền⟩. Quy tắc chọn ảnh nếu chưa gán toàn bộ: ⟨điền⟩.

| Nhãn | Số mẫu | Word Accuracy (%) | Wilson 95% (%) | Micro CER (%) |
|---|---:|---:|---|---:|
| `none` | — | — | — | — |
| `perspective` | — | — | — | — |
| `curved` | — | — | — | — |
| `stylized` | — | — | — | — |
| `mixed` | — | — | — | — |

**Nhận xét:** ⟨điền⟩. Các ảnh chưa gán không được xem là `none`. Phân biệt phối cảnh/chữ cong với kiểu chữ cách điệu.

### 4.6. Độ bền với tác động có kiểm soát

| Tác động | Mức/tham số | Số mẫu | Word Accuracy (%) | Micro CER (%) | Giảm Accuracy so clean (điểm %) | Tăng CER so clean (điểm %) |
|---|---|---:|---:|---:|---:|---:|
| Clean | 0 | — | — | — | 0 theo định nghĩa | 0 theo định nghĩa |
| Gaussian blur | 1/⟨tham số⟩ | — | — | — | — | — |
| Gaussian blur | 2/⟨tham số⟩ | — | — | — | — | — |
| Gaussian blur | 3/⟨tham số⟩ | — | — | — | — | — |
| Perspective | 1/⟨tham số⟩ | — | — | — | — | — |
| Perspective | 2/⟨tham số⟩ | — | — | — | — | — |
| Perspective | 3/⟨tham số⟩ | — | — | — | — | — |
| Wave | 1/⟨tham số⟩ | — | — | — | — | — |
| Wave | 2/⟨tham số⟩ | — | — | — | — | — |
| Wave | 3/⟨tham số⟩ | — | — | — | — | — |

Kèm Wilson của từng mức từ kết quả xuất; có thể để bảng phụ. Vẽ đường Accuracy và CER theo mức tác động. Dùng cùng ảnh và checkpoint; không gộp số mẫu các mức thành số ảnh test độc lập.

| Tác động/mức | Đúng → sai | Sai → đúng | Đúng → đúng | Sai → sai | Delta Accuracy (điểm %) | Bootstrap 95% delta (điểm %) |
|---|---:|---:|---:|---:|---:|---|
| ⟨điền từng mức⟩ | — | — | — | — | — | — |

Ở bảng theo cặp, delta = Accuracy ảnh biến đổi − Accuracy clean; âm là giảm. Mã dùng bootstrap theo cặp 2.000 lần, seed 42 cho khoảng delta, khác Wilson từng nhóm. Cột “giảm Accuracy” ở bảng tác động phía trên là đối dấu của delta này.

**Nhận xét:** ⟨điền xu hướng, trường hợp cải thiện, mức còn đọc được và mức cắt/mất chữ nếu có⟩. Tách kết quả này khỏi nhãn biến dạng tự nhiên.

### 4.7. Ví dụ lỗi và nguyên nhân khả dĩ

| ID ảnh | Nhãn thật | Dự đoán | Loại edit | Độ dài | Chỉ số sắc nét | Nhãn biến dạng | Giải thích có căn cứ |
|---|---|---|---|---:|---:|---|---|
| ⟨điền từ dự đoán thật⟩ | — | — | — | — | — | — | — |

Chèn ảnh để người đọc kiểm tra. Ghi rõ phần giải thích nào là suy luận; không khẳng định một cơ chế gây lỗi chỉ từ một ví dụ. Nêu quy tắc chọn ví dụ, tránh chỉ chọn ảnh thuận cho kết luận.

## Chương 5. Kết luận và hướng phát triển

### 5.1. Kết quả đạt được

⟨Liệt kê những phần thực sự hoàn thành, các chỉ số chính và phát hiện có bằng chứng. Nếu còn thiếu nhãn biến dạng hoặc chưa huấn luyện đủ, nêu rõ.⟩

### 5.2. Giới hạn

⟨Thảo luận quy mô train nhỏ, ngân sách GPU, bảng ký tự, giới hạn chiều rộng, chỉ số sắc nét thay thế, chủ quan gán nhãn, nhóm ít mẫu, seed và khác biệt tác động nhân tạo–tự nhiên. Chỉ nêu giới hạn liên quan đến kết quả thực nghiệm.⟩

### 5.3. Hướng phát triển

⟨Từ bằng chứng lỗi, chọn mở rộng có mục tiêu: dữ liệu tổng hợp/tiền huấn luyện có nguồn rõ; augmentation; khối chỉnh hình cho chữ cong/phối cảnh; decoder cải tiến; hoặc thêm bộ phát hiện chữ để xử lý ảnh toàn cảnh. Không hứa mức cải thiện chưa đo.⟩

## Phụ lục

- Cấu hình và hướng dẫn tái lập.
- Manifest, audit và danh sách mẫu bị loại nếu có.
- Nguồn trọng số/dữ liệu bổ sung và giấy phép nếu sử dụng.
- Ngưỡng mờ train, hướng dẫn/tệp gán nhãn biến dạng.
- Tham số và seed tác động, đường dẫn dự đoán/checkpoint, bảng kết quả đầy đủ.

## Tài liệu tham khảo

Dùng [tai_lieu_tham_khao.bib](tai_lieu_tham_khao.bib). Trích đúng tài liệu đã đọc và phân biệt năm bản arXiv với năm bài báo xuất bản nếu thay nguồn.
