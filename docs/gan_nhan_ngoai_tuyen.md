# Gán nhãn độ mờ và biến dạng từ ảnh gốc

Công cụ đã được xuất tại `data/annotation_tool/index.html`: 200 ảnh được lấy mẫu không hoàn lại từ 3.000 ảnh test, seed 42. Việc chọn mẫu chỉ dùng manifest, không dùng dự đoán hay danh sách lỗi. Ảnh giữ nguyên độ phân giải và được áp dụng hướng EXIF; hai tệp ảnh phân trang được nhúng trong thư mục `chunks`. Công cụ chạy khi mở `index.html` trực tiếp bằng trình duyệt, không cần máy chủ, tài khoản hoặc mạng.

Tệp nhãn chuẩn `data/manifests/test_annotations.csv` hiện vẫn có **0 nhãn độ mờ và 0 nhãn biến dạng**. Các lựa chọn trong giao diện chỉ trở thành dữ liệu dự án sau khi người đánh giá tải CSV và nhập bằng lệnh bên dưới. Không coi ô trống là `clear` hoặc `none`.

## Quan sát và lưu nhãn

1. Mở `data/annotation_tool/index.html` và đọc quy ước trong mục “Quy ước và phím tắt”. Chọn “Xem kích thước gốc” khi cần xem chi tiết nét chữ. Giao diện không hiển thị dự đoán của mô hình.
2. Gán `blur_manual` bằng `clear`, `moderate` hoặc `severe`. Chỉ đánh giá độ mềm/mất chi tiết của nét chữ; không tự suy ra mờ từ độ dài, tương phản hoặc nền nhiều chi tiết.
3. Gán `distortion` bằng `none`, `perspective`, `curved`, `stylized` hoặc `mixed`. Để trống khi chưa đánh giá hoặc chưa chắc và ghi chú trường hợp khó quyết định.
4. Dùng ←/→ để chuyển ảnh; 1/2/3 chọn mức mờ; N/P/C/S/M chọn biến dạng. Phím tắt không hoạt động khi đang nhập ghi chú hoặc chọn một ô.
5. Nhập tên/mã người đánh giá rồi tải cả **CSV tiến độ** và **metadata đánh giá**. Trình duyệt lưu tiến độ theo bộ dữ liệu và bản CSV nguồn, nhưng cần tải định kỳ. Nếu trình duyệt báo không lưu được, CSV vẫn tải được trong phiên hiện tại.

CSV tải xuống có đúng bốn cột `id,blur_manual,distortion,notes` và toàn bộ 3.000 ID theo thứ tự manifest; các ảnh chưa đánh giá giữ ô trống. Nhãn có sẵn trong CSV nguồn được khóa trong giao diện. Giữ thư mục `chunks` cạnh `index.html` khi chuyển công cụ sang máy khác.

## Nhập nhãn đã đánh giá

Chạy tại `C:\HOCSAU`, thay đường dẫn Downloads và mã người đánh giá bằng giá trị thực:

```powershell
.\.venv\Scripts\python.exe -m crnn.annotate import --manifest data/manifests/test.csv --annotations data/manifests/test_annotations.csv --csv "C:\Users\Admin\Downloads\test_annotations_review.csv" --review-metadata "C:\Users\Admin\Downloads\annotation_review_metadata.json" --annotator "reviewer_01" --source human_review
```

Lệnh ghép theo ID và từng trường. Ô trống trong CSV mới nghĩa là giữ dữ liệu cũ, kể cả khi có `--overwrite`. ID không thuộc manifest, ID trùng, nhãn sai quy ước và cấu trúc CSV không hợp lệ đều bị từ chối trước khi ghi. Nhãn cũ khác nhãn mới bị từ chối; chỉ dùng `--overwrite` khi chủ động sửa nhãn đã lưu.

CSV và metadata được ghi bằng thay thế tệp nguyên tử. Sidecar `test_annotations.provenance.json` lưu hash CSV, người đánh giá, nguồn nhãn, thời gian, trường đã đổi và thông tin lấy mẫu nếu cung cấp `--review-metadata`. Công cụ không tự nhận các nhãn có sẵn là nhãn của con người. Nhãn do AI đánh giá phải dùng `--source ai_review` và nên lưu trong CSV riêng để phục vụ phân tích thăm dò, không thay thế tập nhãn được người kiểm tra.

## Tạo bộ quan sát khác

Thư mục xuất phải chưa tồn tại; dùng tên mới khi tạo lại để giữ tiến độ/bản công cụ trước:

```powershell
.\.venv\Scripts\python.exe -m crnn.annotate export --manifest data/manifests/test.csv --annotations data/manifests/test_annotations.csv --output data/annotation_tool_v2 --sample-size 200 --seed 42 --page-size 100
```

Bỏ `--sample-size` để xuất toàn bộ manifest. `selection_metadata.json` ghi danh sách ID đã chọn và hash manifest. Không chọn mẫu dựa trên lỗi nhận dạng khi muốn mô tả phân bố độ mờ/biến dạng của tập test.

## Phân tích nhãn quan sát

```powershell
.\.venv\Scripts\python.exe -m crnn.analyze --predictions runs/baseline_seed42/evaluation/predictions.csv --output runs/baseline_seed42/analysis_manual --annotations data/manifests/test_annotations.csv --calibration data/manifests/train.csv
```

Báo cáo ghi tỷ lệ ảnh đã gán nhãn và nguồn nhãn. Khi có dữ liệu, `manual_blur_groups.csv/.png` mô tả word accuracy, khoảng Wilson 95% và CER micro cho clear/moderate/severe. `manual_blur_proxy_agreement.csv/.png` đối chiếu nhãn quan sát với ba nhóm Laplacian được cố định từ train; summary còn ghi trung vị/IQR điểm proxy, tỷ lệ trùng nhóm, kappa mô tả và Spearman.

Ánh xạ low_sharpness→severe, medium_sharpness→moderate, high_sharpness→clear là quy ước để kiểm tra proxy, không phải bộ phát hiện mờ đã được hiệu chuẩn. Kappa này không đo đồng thuận giữa nhiều người gán nhãn. Nhãn chủ quan cần quy ước nhất quán và nên có một phần ảnh được người thứ hai đánh giá độc lập. Các nhóm quan sát chưa chứng minh quan hệ nhân quả; kết quả từ 200 ảnh cần ghi rõ độ bao phủ và cách chọn mẫu.
