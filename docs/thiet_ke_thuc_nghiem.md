# Thiết kế thực nghiệm và phân tích lỗi

**Trạng thái:** Giao thức thực nghiệm và tiêu chí báo cáo. Kết quả thực chạy được lưu riêng trong [kết quả baseline](ket_qua_baseline.md); cấu hình, manifest và dự đoán từng ảnh là bằng chứng cho các bảng số liệu.

## 1. Dữ liệu và phép so sánh chính

Sử dụng IIIT5K gốc: 2.000 ảnh từ train và 3.000 ảnh từ test. Dữ liệu chứa cả chữ trong cảnh và ảnh born-digital. Phân chia này được mô tả ở mục 3.1 và Bảng 1 của [bài BMVC 2012](https://www.bmva-archive.org.uk/bmvc/2012/BMVC/paper127/paper127.pdf).

Giao thức chính là **huấn luyện từ đầu trên IIIT5K, đánh giá không từ điển**. Không dùng danh sách từ của từng ảnh, nhãn test hay cơ chế sửa từ để giải mã. Cần ghi rõ giao thức vì kết quả dùng từ điển giới hạn 50/1.000 từ không thể so trực tiếp với greedy không từ điển.

CRNN gốc học trên khoảng 8 triệu ảnh tổng hợp và đánh giá các benchmark mà không tinh chỉnh trên phần train của chúng. Thực nghiệm của đề tài phục vụ học kiến trúc với tập nhỏ, không tái lập cùng ngân sách dữ liệu. Nguồn: [Shi, Bai và Yao, mục 3.1](https://arxiv.org/pdf/1507.05717).

## 2. Phân chia và chống rò rỉ

Từ 2.000 mẫu train gốc, tạo train khoảng 1.800 và validation khoảng 200 bằng seed 42. Các ảnh có cùng nhãn đã chuẩn hóa thuộc cùng nhóm; một nhóm chỉ nằm ở train hoặc validation. Nếu chọn được tổng đúng 200 ảnh bằng các nhóm nguyên vẹn thì dùng 1.800/200; nếu không, ghi số thực tế trong metadata và báo cáo. Tập test gốc luôn giữ nguyên.

Nhóm theo từ hạn chế việc một từ giống hệt xuất hiện ở hai phần train/validation. Điều này không chứng minh dữ liệu hoàn toàn độc lập: nhiều từ khác nhau có thể được cắt từ cùng ảnh nguồn, và nhãn từ có thể trùng giữa train gốc và test gốc. Cần báo cáo mức trùng nhãn giữa các phần nếu có; không thay đổi split test chuẩn để che trùng nhãn. Manifest cố định quan trọng hơn việc chia lại bằng seed mỗi lần.

Các lựa chọn sau chỉ dùng train hoặc validation:

| Quyết định | Dữ liệu được phép dùng |
|---|---|
| Học trọng số, tăng cường ảnh train | Train |
| Chọn epoch/checkpoint, learning rate và siêu tham số | Validation |
| Xác định hai ngưỡng phân nhóm chỉ số mờ | Train thực dùng để học |
| Kiểm tra giao thức gán nhãn biến dạng, tham số tác động | Train/validation trước khi khóa giao thức |
| Báo cáo cuối cùng trên test | Checkpoint và giao thức đã khóa |

Nếu điều chỉnh mô hình sau khi xem kết quả test, phải công bố là phân tích khám phá; không gọi kết quả tiếp theo là đánh giá test độc lập.

## 3. Audit trước huấn luyện

Lưu kết quả kiểm tra đường dẫn, số mẫu, ảnh hỏng, nhãn rỗng, bảng ký tự, chiều dài nhãn và số ảnh bị ảnh hưởng bởi giới hạn kích thước. Bảng chữ cái gồm chữ số và tiếng Anh chữ thường; blank là chỉ số 0 và không xuất hiện trong target.

Chuyển nhãn thành chữ thường. Nếu có ký tự ngoài bảng chữ cái, cần dừng để sửa giao thức hoặc công bố mẫu bị loại. Không xóa âm thầm ký tự trong nhãn. Nếu loại mẫu, báo số lượng và lý do theo split; độ chính xác trên tập sau lọc phải đi cùng mẫu số, không được trình bày như độ chính xác trên toàn bộ 3.000 mẫu.

Với mỗi ảnh, kiểm tra \(T_i\ge L_i+r_i\), trong đó \(r_i\) đếm số cặp ký tự giống nhau kề nhau. Nếu không thỏa, cần tăng khả năng biểu diễn hoặc báo rõ cách xử lý trước khi bắt đầu học.

## 4. Tiền xử lý và cấu hình mô hình

Ảnh được đổi thành ảnh xám. Để vừa khung cao 32 và rộng tối đa 320 mà vẫn giữ tỷ lệ, dùng:

\[
s=\min(32/H,320/W),\quad
H'=\operatorname{round}(sH),\quad W'=\operatorname{round}(sW).
\]

Giới hạn kích thước làm tròn tối thiểu 1 pixel. Ảnh thu nhỏ đặt giữa theo chiều dọc trên nền trắng cao 32; padding bên phải để đạt ít nhất 64 pixel và chiều rộng là bội số của 4. Khi ghép batch, padding phải tiếp tục tới chiều rộng lớn nhất của batch, rồi thêm **24 pixel trắng** ở mép phải tensor. Phần này giữ ngữ cảnh nền ở biên CNN nhất quán giữa các cách ghép batch. Giá trị nền và phép chuẩn hóa pixel phải thống nhất trong mọi bước.

Đối với từ rất dài, điều kiện rộng tối đa 320 có thể khiến chiều cao chữ thực tế nhỏ hơn 32. Tiền xử lý giữ tỷ lệ nhưng có thể làm mất chi tiết nét; cần ghi số mẫu này và xem trong phân tích độ dài. Phần padding không phải ký tự; BiLSTM và CTC dùng độ dài hợp lệ từng ảnh. Margin 24 pixel không được tính vào \(W_i\): CNN tạo thêm sáu bước đệm, nhưng packed BiLSTM, CTC và decoder vẫn dùng \(T_i=\lfloor W_i/4\rfloor\). Không dùng các bước margin để làm nhãn thiếu độ dài trở nên khả thi giả tạo.

| Thành phần | Cấu hình giao thức |
|---|---|
| CNN | Kênh 64–128–256–256–512–512–512; pooling giảm rộng tổng hệ số 4 |
| Tích chập cuối | Kernel `(2,1)` để đầu ra cao 1; điều chỉnh so với kernel `2×2` bài gốc |
| Chuỗi CNN | 512 đặc trưng/bước; \(T_i=\lfloor W_i/4\rfloor\) với chiều rộng canvas hợp lệ |
| BiLSTM | Hai tầng, hidden size 256 mỗi hướng, dùng chuỗi đóng gói theo độ dài |
| Đầu ra | Linear + log-softmax; 37 lớp gồm 36 ký tự và blank |
| Loss | CTC với nhãn chuỗi và độ dài hợp lệ; chạy CPU qua tensor giữ gradient |
| Decoder | Greedy CTC không từ điển |
| Seed chính | 42 |

Optimizer, learning rate, batch size, số epoch, dropout, augmentation, precision và lựa chọn checkpoint phải được lấy từ cấu hình thực chạy, không suy từ bảng này. Máy có NVIDIA GPU vẫn cần kiểm tra PyTorch nhận CUDA trước khi báo đã huấn luyện trên GPU. CNN/BiLSTM có thể chạy GPU trong khi `ctc_loss_device=cpu` được ghi trong cấu hình; không mô tả toàn bộ phép tính đều chạy GPU. Chính sách CTC trên CPU hỗ trợ chế độ thuật toán quyết định, có chi phí chuyển dữ liệu cần tính vào thời gian thực đo.

## 5. Thứ tự chạy thực nghiệm

1. Kiểm tra dữ liệu và tạo manifest; lưu cấu hình và audit.
2. Kiểm tra forward, độ dài CTC và loss; học thử tập rất nhỏ để xác nhận mô hình có thể học và decoder hoạt động. Kết quả này chỉ kiểm tra chương trình, không đại diện độ chính xác test.
3. Huấn luyện trên train, đánh giá validation mỗi epoch; lưu đường loss, Word Accuracy và CER.
4. Chọn checkpoint bằng Word Accuracy validation; nếu Accuracy bằng nhau, dùng validation CTC loss thấp hơn để phá hòa. Quy tắc này giúp lựa chọn khi các epoch đầu có thể đều dự đoán blank và Accuracy bằng 0. Chính sách dừng sớm nếu có phải cố định trước khi xem test.
5. Chạy checkpoint đã chọn trên toàn bộ test gốc một lần cho kết quả chính; lưu dự đoán từng ảnh.
6. Tính phân nhóm độ dài và mờ. Gán nhãn biến dạng từ ảnh gốc trước khi xem dự đoán, rồi phân tích.
7. Từ cùng ảnh test tạo các tác động có kiểm soát và đánh giá cùng checkpoint.

Nếu ngân sách cho phép, lặp huấn luyện với các seed trọng số khác và giữ split cố định. Báo trung bình/độ lệch chuẩn giữa lần chạy chỉ khi thực sự đã có nhiều lần chạy. Seed 42 duy nhất không cho phép suy độ biến thiên do khởi tạo.

## 6. Chỉ số và khoảng tin cậy

Với mỗi tập hoặc nhóm, xuất `n`, số từ đúng, Word Accuracy, Wilson 95% và micro CER. Word Accuracy là tỷ lệ nhãn khớp hoàn toàn sau chuẩn hóa. Micro CER là tổng khoảng cách Levenshtein chia tổng ký tự nhãn, không phải trung bình CER các từ.

Với \(\hat p=k/n\), \(z=1.959964\), Wilson 95% có tâm \(c\) và bán kính \(h\):

\[
c=\frac{\hat p+z^2/(2n)}{1+z^2/n},\quad
h=\frac{z}{1+z^2/n}\sqrt{\frac{\hat p(1-\hat p)}n+\frac{z^2}{4n^2}}.
\]

Khoảng là \([c-h,c+h]\). Cơ sở thống kê xem [NIST/SEMATECH, confidence intervals for proportions](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

Với nhóm không có mẫu, ghi `n=0`, chỉ số không xác định; không ghi Accuracy/CER bằng 0. Với nhóm rất ít mẫu, giữ số lượng và diễn giải dè dặt. Wilson mô tả bất định của tỷ lệ theo giả định mẫu Bernoulli độc lập; không bao quát biến thiên giữa seed hay phụ thuộc của các vùng cắt cùng nguồn. Khoảng tin cậy của Word Accuracy không phải khoảng tin cậy của CER.

## 7. Phân tích theo độ dài từ

Tính độ dài từ **nhãn thật sau chuẩn hóa**, không từ dự đoán:

| Nhóm | Độ dài nhãn |
|---|---|
| `1-3` | 1 đến 3 |
| `4-6` | 4 đến 6 |
| `7-9` | 7 đến 9 |
| `10+` | Từ 10 |

Vẽ Word Accuracy kèm Wilson và CER theo nhóm; kèm `n`. Kiểm tra thêm lỗi xóa ký tự, chữ lặp, \(T_i/(L_i+r_i)\) và số mẫu bị giới hạn rộng. Nếu Word Accuracy giảm còn CER gần ổn định, giải thích bằng tiêu chí đúng toàn từ, không kết luận mô hình hỏng hoàn toàn ở từ dài.

## 8. Phân tích độ mờ trên dữ liệu gốc

Định nghĩa chỉ số sắc nét thay thế:

\[
q(I)=\operatorname{Var}(\nabla^2 I_{64}),
\]

trong đó \(I_{64}\) là ảnh gốc xám được đổi tỷ lệ để cao 64, giữ tỷ lệ ngang, **không có canvas padding**. Triển khai dùng nội suy bicubic, thang xám 0–255, toán tử Laplacian bốn láng giềng (tổng trên/dưới/trái/phải trừ bốn lần điểm giữa), bỏ biên một pixel trước khi lấy phương sai. Chuẩn hóa kích thước giúp phép so sánh không chỉ phản ánh ảnh vốn lớn hay nhỏ; vẫn cần cùng nội suy, thang pixel và toán tử Laplacian cho mọi ảnh.

Tính hai ngưỡng phân vị \(1/3\) và \(2/3\), tức khoảng 33,33% và 66,67%, trên train thực dùng để học, rồi đóng băng:

| Nhóm | Quy tắc với hai ngưỡng \(a\le b\) |
|---|---|
| `low_sharpness` | \(q\le a\) |
| `medium_sharpness` | \(a<q\le b\) |
| `high_sharpness` | \(q>b\) |

Không tính lại phân vị trên test để chia đều số ảnh. Giá trị trùng ngưỡng có thể làm các nhóm không bằng nhau. Thuật ngữ “mờ” trong báo cáo phải đi cùng mô tả đây là chỉ số thay thế: ảnh nền nhiều cạnh, nhiễu, tương phản thấp hoặc chữ có nét đơn giản đều có thể ảnh hưởng \(q\). Chọn vài ảnh mỗi nhóm và đối chiếu thị giác trước khi diễn giải kết quả.

## 9. Gán nhãn biến dạng tự nhiên

IIIT5K không cung cấp sẵn nhãn chuẩn cho các nhóm biến dạng bên dưới. Phân tích này cần gán nhãn thủ công; không dùng tỷ lệ khung ảnh để tự kết luận ảnh có phối cảnh hoặc chữ cong.

| Nhãn | Quy tắc gán nhãn |
|---|---|
| `none` | Chưa thấy biến dạng hình học hoặc kiểu chữ khác thường đáng kể. Không có nghĩa ảnh dễ đọc. |
| `perspective` | Chữ nằm trên mặt phẳng nhìn nghiêng, có hội tụ hoặc kích thước nét thay đổi theo phối cảnh. |
| `curved` | Đường sắp xếp/baseline của chữ cong, uốn hoặc bố trí theo cung. |
| `stylized` | Nét và hình dạng ký tự được thiết kế cách điệu đáng kể, dù baseline có thể thẳng. |
| `mixed` | Hai hoặc nhiều loại trên xuất hiện rõ ràng trong cùng ảnh. |

Đây là quy ước vận hành của đề tài; kiểu chữ cách điệu và biến dạng hình học là hai hiện tượng khác nhau được đưa vào chung bảng phân tích để trả lời yêu cầu “biến dạng chữ”. Các ảnh chưa xét giữ trạng thái chưa gán nhãn; không tự đổi thành `none`.

Xem ảnh gốc không kèm nhãn dự đoán/đúng sai. Mẫu `test_annotations.csv` có cột `id`, `blur_manual`, `distortion`, `notes`; tra ID trong manifest để mở ảnh. Điền `distortion` theo bảng; `blur_manual` và `notes` hỗ trợ kiểm tra bằng mắt, chưa thay thế chỉ số mờ chính. Nên có hai người xét một phần ảnh và thống nhất bất đồng; nếu chỉ một người, công bố điều đó. Ghi hướng dẫn, người gán, ngày, số ảnh đã gán và mức bất đồng nếu có. Không gán nhãn chỉ cho ảnh mô hình nhận sai, vì sẽ làm sai mẫu số nhóm.

Nếu chưa gán toàn bộ test, báo rõ độ phủ và cách chọn mẫu. Chỉ phân tích tập đã gán; không khẳng định tỷ lệ đó đại diện toàn bộ IIIT5K. Nhóm có ít mẫu chỉ hỗ trợ nhận xét định tính.

## 10. Tác động có kiểm soát và so sánh theo cặp

Thử nghiệm dùng ảnh test gốc làm mức `clean`, sau đó tạo độc lập:

| Tác động | Ý nghĩa | Mức |
|---|---|---|
| `gaussian_blur` | Làm mờ bằng Gaussian | Bán kính 0,7; 1,2; 2,0 pixel trên ảnh gốc |
| `perspective` | Biến đổi phối cảnh | Biên ngang 4%; 8%; 12% và biên dọc 5%; 10%; 15% |
| `wave` | Biến dạng uốn sóng một chu kỳ | Biên độ bằng 4%; 7%; 10% của `(H−1)` |

Mức 1–3 biểu thị tham số tăng dần trong chương trình, không phải mức mờ/biến dạng tự nhiên đã hiệu chuẩn. Các góc phối cảnh dùng hệ số biên cố định; wave thu nhỏ dọc trước khi dịch để giữ toàn bộ ảnh trong canvas. Vì vậy wave thay đổi cả đường chữ lẫn kích thước dọc, không phải phép can thiệp vào độ cong đơn thuần. Nền điền trắng; phối cảnh nội suy bicubic và wave nội suy tuyến tính dọc. Chi tiết nằm trong `CORRUPTION_SPEC` ở `src/crnn/distortions.py` và được giữ với kết quả đánh giá.

Tác động thực hiện trên ảnh gốc trước tiền xử lý CRNN, có kích thước canvas gốc. Gaussian có bán kính pixel cố định nên mức tác động sau resize có thể khác nhau theo độ phân giải ban đầu. Lưu tham số, seed nếu có thành phần ngẫu nhiên, phiên bản phép biến đổi và thứ tự tiền xử lý. Dùng tham số như nhau cho mọi mô hình đem so sánh; kiểm tra montage ở mỗi mức để xác nhận nhãn vẫn có ý nghĩa sau tác động.

Mỗi ảnh biến đổi có cùng ID và nhãn với ảnh gốc. Không huấn luyện lại theo từng mức, không lấy ảnh biến đổi test làm train. Các tác động được chạy riêng từ ảnh gốc; không vô tình áp dụng liên tiếp Gaussian rồi phối cảnh rồi wave.

Với từng tác động/mức, báo Word Accuracy, Wilson, CER và:

\[
\Delta\operatorname{Acc}=\operatorname{Acc}_{corrupt}-\operatorname{Acc}_{clean}
\quad\text{(điểm phần trăm nếu nhân 100)},
\]

\[
\Delta\operatorname{CER}=\operatorname{CER}_{corrupt}-\operatorname{CER}_{clean}.
\]

Theo quy ước đầu ra `delta_word_accuracy`, delta âm biểu thị giảm Accuracy; nếu báo “mức giảm” dương thì lấy \(-\Delta\operatorname{Acc}\). Bảng theo cặp đếm `đúng → sai`, `sai → đúng`, `đúng → đúng`, `sai → sai`. Chương trình ước lượng khoảng 95% cho delta Accuracy bằng bootstrap theo cặp 2.000 lần, seed 42, lấy phân vị 2,5% và 97,5%; các ảnh cùng ID được giữ cặp khi lấy mẫu lại. Đây là bootstrap phần trăm theo ảnh, không phải Wilson và không đo bất định giữa các lần huấn luyện.

Một mức có thể cải thiện một số ảnh; không ép đường kết quả giảm đơn điệu. Các phiên bản cùng ảnh có phụ thuộc, nên không gộp mọi mức thành một tập lớn để coi là nhiều ảnh test độc lập. Wilson từng mức không phải khoảng tin cậy của chênh lệch theo cặp.

Kết quả tác động nhân tạo và nhóm biến dạng tự nhiên được báo riêng: chúng trả lời hai câu hỏi khác nhau. Không dùng thí nghiệm wave để khẳng định đã đo tỷ lệ lỗi trên mọi chữ cong trong đời thực.

## 11. Phân tích lỗi định tính và diễn giải

Chọn ảnh đúng/sai theo quy tắc đã ghi, kèm ID, ảnh, nhãn thật, dự đoán, độ dài, chỉ số mờ, nhãn biến dạng và edit distance. Phân loại các ví dụ: thay ký tự tương tự, xóa ký tự, thêm ký tự, mất chữ lặp, sai nhiều ký tự hoặc đầu ra rỗng. Các cặp như `0/o`, `1/l`, `rn/m` là ví dụ giả định để kiểm tra, chỉ đưa vào kết luận khi kết quả thật có chúng.

Phân nhóm riêng lẻ không loại bỏ yếu tố nhiễu. Có thể bổ sung bảng giao độ dài × mờ khi đủ mẫu, hoặc xem ảnh tiêu biểu ở độ dài tương đương. Nhóm ít mẫu không phù hợp cho kết luận mạnh. Không gọi mối liên hệ nhóm là quan hệ nhân quả.

## 12. Thí nghiệm bổ sung có điều kiện

- **CRNN đã huấn luyện:** cần nguồn và dữ liệu tiền huấn luyện rõ; dùng cùng split, nhãn, decoder và chỉ số. Không so sánh nếu alphabet hoặc cách chọn ảnh khác nhau mà chưa giải thích.
- **Augmentation:** nếu bổ sung mô hình có augmentation train, lưu cấu hình riêng và chỉ chọn bằng validation. So trên cùng ảnh test gốc cùng tác động để kiểm tra độ bền.
- **Beam search/rectification:** là mở rộng sau baseline; ghi tác động của decoder hoặc khối chỉnh hình, không gộp mọi thay đổi rồi quy cải thiện cho một thành phần.

Chỉ báo các thí nghiệm thực sự đã chạy. Thiết kế này không cam kết mô hình tiền huấn luyện hoặc augmentation nhất định sẽ cải thiện điểm số.

## 13. Hồ sơ tái lập cần giữ

Giữ nguồn và phiên bản IIIT5K, hash/tên tệp dữ liệu, manifest các split, bảng ký tự, seed, cấu hình tiền xử lý, số mẫu audit, cấu hình huấn luyện, phiên bản Python/PyTorch/CUDA, GPU, lịch sử mỗi epoch, checkpoint được chọn và lý do chọn. Với phân tích, giữ ngưỡng mờ train, tệp gán nhãn, tham số tác động và dự đoán từng ảnh.

Các tệp kết quả phải liên kết được về cùng checkpoint và manifest. Không chép biểu đồ từ một lần chạy khác vào báo cáo của lần chạy hiện tại.

| Giai đoạn | Tệp đầu ra chính hiện có trong chương trình |
|---|---|
| Chuẩn bị | `train.csv`, `val.csv`, `test.csv`, `split_metadata.json`, `test_annotations.csv` |
| Huấn luyện | `config.json`, `history.csv`, `best.pt` |
| Đánh giá mỗi điều kiện | `predictions.csv`, `metrics.json`, `evaluation_metadata.json` |
| Phân tích | `summary.json`, `report.md`, `errors.csv`, các bảng/PNG phân nhóm; `paired_robustness.csv` khi có cặp |

`evaluation_metadata.json` ghi hash checkpoint, manifest và quy ước giải mã; phân tích theo cặp kiểm tra ID/nhãn và tính nhất quán của provenance. Gán nhãn tự nhiên còn cần người thực hiện; bộ mã không tự tạo nhãn đúng thay cho bước này.
