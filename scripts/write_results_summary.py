"""Write a Vietnamese experiment summary from actual saved artifacts."""

import argparse
import csv
import json
from pathlib import Path

from crnn.metrics import summarize_predictions

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def group_table(groups):
    lines = ["| Nhóm | n | Word Accuracy | Wilson 95% | CER micro |",
             "|---|---:|---:|---|---:|"]
    for row in groups:
        lines.append(f"| {row['group']} | {row['num_samples']} | {pct(row['word_accuracy'])} | "
                     f"{pct(row['word_accuracy_ci_low'])}–{pct(row['word_accuracy_ci_high'])} | {pct(row['cer'])} |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/baseline_seed42")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/ket_qua_baseline.md")
    args = parser.parse_args()
    run = args.run.resolve()
    relative = run.relative_to(ROOT).as_posix()
    config = load(run / "config.json")
    summary = load(run / "analysis/summary.json")
    metrics = load(run / "evaluation/metrics.json")
    with (run / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    overall = summary["overall"]
    with Path(config['manifests']['train']['path']).open(encoding="utf-8-sig", newline="") as handle:
        seen_labels = {row["label"].lower() for row in csv.DictReader(handle)}
    with (run / "evaluation/predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    seen_groups = []
    for name, seen in [("seen_in_train", True), ("unseen_in_train", False)]:
        rows = [row for row in predictions if (row["label"] in seen_labels) == seen]
        if rows:
            seen_groups.append({"group": name, **summarize_predictions(rows)})
    with (run / "analysis/seen_word_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seen_groups[0]))
        writer.writeheader()
        writer.writerows(seen_groups)
    epoch = metrics["evaluation"]["checkpoint_epoch"]
    selected = next(row for row in history if int(row["epoch"]) == epoch)
    lines = ["# Kết quả baseline CRNN trên IIIT5K", "",
             "**Kết quả thực chạy**, lấy từ checkpoint, dự đoán và lịch sử huấn luyện lưu trong dự án. "
             "Đây là baseline đầu tiên cho đề tài; chưa có so sánh nhiều seed hoặc mô hình tiền huấn luyện.", "",
             "## Cấu hình và nguồn số liệu", "",
             f"- Run: `{relative}`; seed trọng số: {config['seed']}.",
             f"- Train: {config['manifests']['train']['samples']} ảnh, validation: {config['manifests']['val']['samples']} ảnh; "
             f"test: {overall['num_samples']} ảnh. Các từ giống nhau sau chuẩn hóa không chia qua train/validation.",
             f"- Hoàn thành {len(history)} epoch; checkpoint chọn ở epoch **{epoch}** theo Word Accuracy validation "
             "và loss thấp hơn khi bằng điểm.",
             f"- Validation tại checkpoint: Word Accuracy **{pct(float(selected['val_word_accuracy']))}**, "
             f"CER **{pct(float(selected['val_cer']))}**.",
             f"- PyTorch: {config['torch_version']}; CNN/BiLSTM trên {config['device']}, CTC loss trên {config['ctc_loss_device']}.",
             f"- Batch {config['arguments']['batch_size']}, Adam lr={config['arguments']['lr']}; "
             f"BiLSTM hai tầng, hidden {config['model_config']['hidden_size']} mỗi hướng, 36 ký tự và blank, greedy không từ điển.",
             f"- Padding batch: `{config['batch_padding']}`; giữ tỷ lệ ảnh và độ dài hợp lệ, thêm ngữ cảnh trắng cho biên CNN.",
             f"- Thời gian cộng từ lịch sử: {sum(float(row['seconds']) for row in history):.1f} giây. "
             "Đây là thời gian quan sát trên máy hiện tại có thể có tác vụ khác, không phải benchmark tốc độ chuẩn.",
             f"- Nguồn: [config](../{relative}/config.json), [môi trường](../{relative}/environment.json), "
             f"[history](../{relative}/history.csv), [metadata đánh giá](../{relative}/evaluation/evaluation_metadata.json).", "",
             "## Đánh giá trên ảnh test gốc", "",
             f"Word Accuracy: **{pct(overall['word_accuracy'])}** ({overall['num_correct']}/{overall['num_samples']} từ). "
             f"Wilson 95%: **{pct(overall['word_accuracy_ci_low'])}–{pct(overall['word_accuracy_ci_high'])}**.", "",
             f"CER micro: **{pct(overall['cer'])}**, tương ứng {overall['character_edits']} phép sửa trên "
             f"{overall['num_characters']} ký tự nhãn thật. CER không phải tỷ lệ từ sai.", ""]
    train_metrics_path = run / "evaluation_train/metrics.json"
    if train_metrics_path.exists():
        train_metrics = load(train_metrics_path)
        lines += [f"Cùng checkpoint đạt Word Accuracy **{pct(train_metrics['word_accuracy'])}** và CER "
                  f"**{pct(train_metrics['cer'])}** trên train khi đánh giá ở chế độ suy luận. "
                  "Khoảng cách với test gợi ý mô hình học sát dữ liệu train nhưng tổng quát hóa còn hạn chế; "
                  "khác biệt phân phối giữa các phần dữ liệu cũng có thể góp phần.", ""]
    lines += ["## Phân tích bổ sung về từ đã gặp", "",
              "Phân tích thăm dò sau khi đọc kết quả baseline; không dùng để chọn checkpoint. "
              "Các ảnh vẫn là test gốc, phân nhóm bằng nhãn từ đã/ chưa xuất hiện trong train.", "",
              "| Nhãn từ | n | Word Accuracy | CER |", "|---|---:|---:|---:|"]
    for row in seen_groups:
        lines.append(f"| {row['group']} | {row['num_samples']} | {pct(row['word_accuracy'])} | {pct(row['cer'])} |")
    lines += ["", "Chênh lệch hai nhóm không tự chứng minh cơ chế ghi nhớ; độ dài và phân phối ảnh cũng khác nhau. "
              "Validation chia theo nhóm từ còn test giữ split gốc, vì vậy không coi độ khó hai phần là giống nhau.", "",
              "## Theo độ dài nhãn", "", *group_table(summary["length_groups"]), "",
              "Đây là kết quả phân tầng quan sát của lần chạy này. Word Accuracy yêu cầu đúng toàn bộ từ, "
              "nên cần xem cùng CER và số mẫu trước khi giải thích tác động của độ dài.", "",
              "## Theo chỉ số sắc nét", ""]
    blur = summary["blur_analysis"]
    if blur["enabled"]:
        lines += [f"Ngưỡng được tính từ train: **{blur['thresholds'][0]:.4f} / {blur['thresholds'][1]:.4f}**, "
                  "phân vị 1/3 và 2/3. Điểm là phương sai Laplacian ảnh xám cao 64 pixel, không dùng padding.", "",
                  *group_table(blur["groups"]), "",
                  "`low_sharpness` nghĩa chỉ số thấp, chưa phải nhãn mờ chuẩn. Font, nền, nhiễu và tương phản "
                  "có thể ảnh hưởng điểm; bảng này không chứng minh quan hệ nhân quả.", ""]
        low_group, _, high_group = blur["groups"]
        if low_group["num_samples"] and high_group["num_samples"]:
            lines += [f"Trong lần chạy này, accuracy nhóm điểm thấp là {pct(low_group['word_accuracy'])}, "
                      f"nhóm điểm cao là {pct(high_group['word_accuracy'])}. "
                      "Nếu nhóm điểm cao kém hơn, phân tầng này không hỗ trợ giả thuyết đơn giản “mờ hơn thì lỗi nhiều hơn”. "
                      "Cần kiểm tra nhãn mờ trực quan và xem riêng tác động blur có kiểm soát bên dưới.", ""]
    lines += ["## Mờ và biến dạng có kiểm soát", "",
              "Cùng các ID ảnh test, cùng nhãn, checkpoint, tiền xử lý và decoder; mỗi biến đổi được đánh giá riêng. "
              "Delta accuracy = biến đổi − gốc, đơn vị điểm phần trăm. Delta CER cũng bằng biến đổi − gốc.", "",
              "| Biến đổi | Mức | Accuracy | Delta accuracy (đpt) | CER | Delta CER (đpt) |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in summary["paired_comparisons"]:
        lines.append(f"| {row['condition']} | {row['severity']} | {pct(row['corrupted_word_accuracy'])} | "
                     f"{100 * row['delta_word_accuracy']:+.2f} | {pct(row['corrupted_cer'])} | {100 * row['delta_cer']:+.2f} |")
    lines += ["", "Khoảng tin cậy bootstrap theo cặp và số chuyển đúng→sai/sai→đúng nằm trong "
              f"[báo cáo chi tiết](../{relative}/analysis/report.md). Biến đổi hình học còn có nội suy và thu nhỏ "
              "để giữ chữ trong canvas; không coi tác động này là đại diện đầy đủ cho mọi biến dạng tự nhiên.", "",
              "## Biến dạng tự nhiên và giới hạn", "",
              f"Hiện có **{summary['num_distortion_annotations']}** ảnh test được gán nhãn biến dạng tự nhiên. "
              "Nếu số này bằng 0, chưa có kết quả phân tầng biến dạng tự nhiên; các ô trống không được tính là `none`. "
              "Cần gán nhãn theo [hướng dẫn](gan_nhan.md) để bổ sung phần này.", "",
              "Baseline chỉ học từ đầu trên 1.800 ảnh IIIT5K. Bài CRNN gốc huấn luyện trên khoảng 8 triệu ảnh tổng hợp; "
              "hai kết quả khác giao thức và ngân sách dữ liệu, không thể dùng điểm baseline để kết luận tái lập bài gốc. "
              "Nguồn: [Shi, Bai và Yao](https://arxiv.org/pdf/1507.05717).", "",
              "Đây là một lần chạy với một seed. Wilson mô tả bất định tỷ lệ theo giả định ảnh độc lập; "
              "không đo biến thiên giữa các lần khởi tạo. Giữ dữ liệu test cho đánh giá và công bố việc đã xem kết quả "
              "nếu tiếp tục điều chỉnh mô hình trên cùng benchmark.", "",
              "Các bước tiếp theo cho đề tài: gán nhãn biến dạng tự nhiên, kiểm tra trực quan các nhóm mờ, "
              "thử tiền huấn luyện có nguồn rõ hoặc augmentation với lựa chọn bằng validation; báo riêng từng giao thức.", "",
              "## Biểu đồ và ví dụ lỗi", "",
              f"![Đường học](../{relative}/figures/training_curves.png)", "",
              f"![Độ bền](../{relative}/figures/robustness_curves.png)", "",
              f"![Ảnh lỗi](../{relative}/figures/error_examples.png)", "",
              f"![Các tác động](../{relative}/figures/corruption_examples.png)", "",
              f"Bảng đầy đủ và dữ liệu từng ảnh: [analysis](../{relative}/analysis/report.md), "
              f"[predictions](../{relative}/evaluation/predictions.csv).", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote measured experiment summary: {args.output}")


if __name__ == "__main__":
    main()
