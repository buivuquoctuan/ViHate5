# Hướng dẫn chạy code trên máy hiện tại

Dự án ở `C:\HOCSAU`, dùng PowerShell và Python trong `.venv`. Dữ liệu IIIT5K, môi trường và checkpoint đã có sẵn. Để thử nhận dạng, thực hiện bước 1 và 2; để huấn luyện lại, làm thêm bước 5. Không cần chạy lại bước tải dữ liệu trên máy này.

## 1. Mở Terminal và kiểm tra môi trường

Trong VS Code chọn **Terminal → New Terminal**, dùng PowerShell:

```powershell
Set-Location C:\HOCSAU
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Máy hiện tại trả về PyTorch `2.5.1+cu121`, `CUDA: True`, RTX 3050 Laptop. Trong VS Code, chọn **Python: Select Interpreter → `.venv\Scripts\python.exe`**. Các lệnh bên dưới gọi executable trực tiếp nên không cần activate môi trường.

Kiểm tra mã:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tệp trong `tests/` là kiểm thử, không phải chương trình huấn luyện. Chạy riêng một tệp bằng `python -m pytest tests/test_model.py -q`, thay `python` bằng đường dẫn `.venv` như trên.

## 2. Nhận dạng ngay bằng mô hình đã học

Checkpoint chính là `runs/improved_seed42/best.pt`, chọn bằng validation; Accuracy test đã lưu là **62,10%**. Mô hình đọc vùng ảnh chứa một từ, với bảng ký tự `0–9, a–z` và không phân biệt hoa/thường.

Thử hai ảnh có sẵn:

```powershell
.\.venv\Scripts\python.exe -m crnn.predict --checkpoint runs/improved_seed42/best.pt --image data/raw/IIIT5K/test/1002_1.png data/raw/IIIT5K/test/1002_2.png --device cuda
```

Kết quả đã kiểm tra trên GPU: `private` và `parking`. Để thử ảnh của bạn, thay đường dẫn sau `--image`:

```powershell
.\.venv\Scripts\python.exe -m crnn.predict --checkpoint runs/improved_seed42/best.pt --image "C:\duong_dan\anh_tu.png" --device cuda
```

Ảnh cần được cắt chứa một từ. Nếu chạy trên máy không có CUDA, dùng `--device cpu` hoặc `--device auto`. Thêm `--output runs/my_predictions.json` nếu muốn lưu JSON.

## 3. Đánh giá lại toàn bộ 3.000 ảnh test

Mỗi lần đánh giá dùng thư mục mới. Đoạn dưới tự tạo tên theo thời gian; chạy các lệnh trong cùng Terminal để giữ biến `$Eval`.

```powershell
$Eval = 'runs/evaluate_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m crnn.evaluate --checkpoint runs/improved_seed42/best.pt --manifest data/manifests/test.csv --output $Eval --batch-size 16 --device cuda
```

Trong `$Eval` có `metrics.json` (Word Accuracy/CER), `predictions.csv` (nhãn và dự đoán từng ảnh), `evaluation_metadata.json` (nguồn checkpoint và cấu hình). Nếu thư mục đã có kết quả, chương trình dừng và yêu cầu chọn tên mới.

## 4. Phân tích lỗi

Dùng kết quả của bước 3:

```powershell
$Analysis = Join-Path $Eval 'analysis'
.\.venv\Scripts\python.exe -m crnn.analyze --predictions "$Eval/predictions.csv" --output $Analysis --calibration data/manifests/train.csv --annotations data/manifests/test_annotations.csv
.\.venv\Scripts\python.exe -m crnn.error_patterns --predictions "$Eval/predictions.csv" --output $Analysis --train-manifest data/manifests/train.csv
```

Mở `$Analysis/report.md`. Các bảng/biểu đồ gồm nhóm độ dài, chỉ số sắc nét, danh sách lỗi; `error_patterns.json` và `character_confusions.csv` phân tích thay/xóa/thêm ký tự. Laplacian là chỉ số thay thế; phân tích theo nhãn mờ/biến dạng tự nhiên chỉ có khi đã gán nhãn.

Để đọc kết quả đã có, mở [phân tích mô hình chính](../runs/improved_seed42/analysis/report.md), [so sánh ba seed](../runs/comparison/comparison.md) hoặc [báo cáo đề tài](bao_cao.md). Không cần đánh giá lại để xem các tệp này.

## 5. Huấn luyện một lần chạy mới

### Baseline từ đầu

```powershell
$Run = 'runs/baseline_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m crnn.train --train data/manifests/train.csv --val data/manifests/val.csv --output $Run --epochs 50 --batch-size 16 --seed 42 --device cuda
```

Mã lưu `config.json`, `history.csv`, `training_summary.json`, `best.pt`. Checkpoint chọn theo Accuracy validation, hòa thì loss thấp hơn. Seed huấn luyện không thay manifests: split hiện tại luôn giữ 1.800 train/200 validation/3.000 test.

Đánh giá và phân tích lần chạy vừa tạo:

```powershell
.\.venv\Scripts\python.exe -m crnn.evaluate --checkpoint "$Run/best.pt" --manifest data/manifests/test.csv --output "$Run/evaluation" --batch-size 16 --device cuda
.\.venv\Scripts\python.exe -m crnn.analyze --predictions "$Run/evaluation/predictions.csv" --output "$Run/analysis" --calibration data/manifests/train.csv --annotations data/manifests/test_annotations.csv
.\.venv\Scripts\python.exe -m crnn.error_patterns --predictions "$Run/evaluation/predictions.csv" --output "$Run/analysis" --train-manifest data/manifests/train.csv
.\.venv\Scripts\python.exe scripts/visualize_results.py --run $Run
```

Hoặc chạy baseline → đánh giá → phân tích → hình → tóm tắt bằng một lệnh; script tự chọn tên mới:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1 -SkipDownload -Epochs 50 -BatchSize 16 -Device cuda
```

Thêm `-Robustness` để đánh giá đủ chín điều kiện blur/perspective/wave. Nếu chỉ chạy thử kỹ thuật, dùng 1–2 epoch và không gọi kết quả đó là baseline đủ 50 epoch.

### Tinh chỉnh từ checkpoint tổng hợp

Máy đã có `runs/synthetic_pretrain_v1/best.pt`. Để thử lại công thức cải tiến:

```powershell
$FineRun = 'runs/finetune_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m crnn.train --train data/manifests/train.csv --val data/manifests/val.csv --output $FineRun --init-checkpoint runs/synthetic_pretrain_v1/best.pt --epochs 50 --batch-size 16 --seed 42 --lr 0.0003 --optimizer adamw --weight-decay 0.0001 --dropout 0.2 --augmentation light --scheduler plateau --early-stopping-patience 12 --device cuda
```

Dùng `$FineRun` thay `$Run` trong các lệnh đánh giá/phân tích phía trên. `--init-checkpoint` nạp trọng số và tạo optimizer mới; không phải tiếp tục optimizer từ một lần train bị ngắt. CNN/BiLSTM chạy GPU, CTC loss chạy CPU và vẫn truyền gradient.

## 6. Gán nhãn mờ và biến dạng

Mở công cụ đã tạo:

```powershell
Invoke-Item data/annotation_tool/index.html
```

Xem 200 ảnh gốc, gán mức mờ và biến dạng, tải CSV cùng metadata rồi nhập theo [hướng dẫn gán nhãn](gan_nhan_ngoai_tuyen.md). Các lựa chọn `clear` và `none` đã được sửa và kiểm tra trong trình duyệt. Nhãn chuẩn hiện còn trống; không xem ảnh chưa gán là không biến dạng.

## 7. Cài mới trên máy khác

Phần này dành cho máy chưa có môi trường/dữ liệu. Cài Python 3.12, sao chép mã và chạy trong thư mục dự án, thay đường dẫn Python bằng executable thực tế:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -PythonPath 'C:\Path\To\Python\python.exe'
.\.venv\Scripts\python.exe -m crnn.download
.\.venv\Scripts\python.exe -m crnn.prepare --root data/raw/IIIT5K --output data/manifests --seed 42
```

Nếu dùng CPU, thêm `-CpuOnly` vào setup. Nếu mang dữ liệu/manifests hiện tại sang, giữ nguyên đường dẫn tương đối và bỏ bước chuẩn bị lại. Thư mục `.venv` cần tạo lại trên máy khác.

Để tái lập cả đợt học trước và nhiều seed, cần baseline seed 42 trước, rồi làm theo [mục thí nghiệm cải tiến trong README](../README.md). Bộ tổng hợp dùng font Windows; thư mục sinh phải mới. Driver `run_extended.py` của workspace hiện tại đã hoàn tất, không cần khởi động lại để nhận dạng ảnh.

## Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `No module named crnn` | Dùng Python trong `.venv`; nếu môi trường mới, chạy `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` |
| `Output already contains a run` / `Evaluation output already exists` | Đổi tên thư mục đầu ra; không xóa checkpoint đang dùng để tiếp tục lệnh |
| `CUDA requested...` | Kiểm tra bước 1; chọn `--device cpu` khi môi trường không dùng CUDA |
| `CUDA out of memory` | Giảm `--batch-size` xuống 8 hoặc 4 và ghi cấu hình mới |
| Không tìm thấy ảnh | Kiểm tra đường dẫn sau `--image`, đặt trong dấu ngoặc kép nếu có khoảng trắng |
| `Unsupported character` | Nhãn huấn luyện cần đúng bảng `0–9, a–z`; mô hình này chưa hỗ trợ đầy đủ tiếng Việt |
| Run bị ngắt giữa epoch | Giữ artifact cũ để đối chiếu và tạo run mới; `run_extended.py --resume` chỉ bỏ qua stage hoàn tất, không phục hồi optimizer |

Kiểm tra mã và danh sách tệp đã dọn: [kiểm tra code](kiem_tra_code.md).
