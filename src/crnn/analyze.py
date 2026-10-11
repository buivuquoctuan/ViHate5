"""Analyze recognition errors without fitting analysis bins on test outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import edit_distance, summarize_predictions
from .distortions import CORRUPTION_KINDS
from .annotate import BLUR_TAGS, annotation_provenance_path, read_annotation_rows

LENGTH_GROUPS = ("1-3", "4-6", "7-9", ">=10")
DISTORTION_GROUPS = ("none", "perspective", "curved", "stylized", "mixed")
BLUR_GROUPS = ("low_sharpness", "medium_sharpness", "high_sharpness")
PROVENANCE_KEYS = ("checkpoint", "checkpoint_sha256", "model_id", "decoder", "normalization", "preprocessing", "alphabet", "batch_padding")


def length_group(length: int) -> str:
    if length < 1:
        raise ValueError("Ground-truth words must contain at least one character")
    return "1-3" if length <= 3 else "4-6" if length <= 6 else "7-9" if length <= 9 else ">=10"


def wilson_interval(correct: int, total: int) -> tuple[float | None, float | None]:
    """Two-sided 95% Wilson score interval for word accuracy."""
    if total < 0 or not 0 <= correct <= total:
        raise ValueError("Invalid counts for a binomial confidence interval")
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def group_summary(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows:
        metrics = summarize_predictions(rows)
    else:
        metrics = {"num_samples": 0, "num_correct": 0, "num_errors": 0,
                   "word_accuracy": None, "cer": None, "character_edits": 0,
                   "num_characters": 0, "mean_normalized_edit_distance": None}
    low, high = wilson_interval(metrics["num_correct"], metrics["num_samples"])
    return {"group": name, **metrics, "word_accuracy_ci_low": low, "word_accuracy_ci_high": high}


def read_predictions(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"id", "label", "prediction"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        seen: set[str] = set()
        for number, raw in enumerate(reader, 2):
            row: dict[str, Any] = dict(raw)
            sample_id = row["id"]
            if not sample_id or sample_id in seen:
                raise ValueError(f"{path}:{number}: missing or duplicate id {sample_id!r}")
            seen.add(sample_id)
            if row["label"] is None or row["prediction"] is None or not row["label"]:
                raise ValueError(f"{path}:{number}: missing prediction or empty label")
            row["length"] = len(row["label"])
            row["edit_distance"] = edit_distance(row["label"], row["prediction"])
            row["correct"] = row["label"] == row["prediction"]
            row["mean_normalized_edit_distance"] = row["edit_distance"] / max(row["length"], len(row["prediction"]), 1)
            if row.get("blur_score") not in (None, ""):
                score = float(row["blur_score"])
                if not math.isfinite(score) or score < 0:
                    raise ValueError(f"{path}:{number}: invalid blur_score")
                row["blur_score"] = score
            else:
                row["blur_score"] = None
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty prediction file")
    return rows


def read_provenance(path: Path | str, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Collect comparable evaluation provenance, if the exporter provides it."""
    result: dict[str, str] = {}
    for key in PROVENANCE_KEYS:
        values = {str(row[key]) for row in rows if row.get(key) not in (None, "")}
        if len(values) > 1:
            raise ValueError(f"{path}: inconsistent {key} within prediction file")
        if values:
            result[key] = next(iter(values))
    metadata_path = Path(path).parent / "evaluation_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in PROVENANCE_KEYS:
            if metadata.get(key) is not None:
                value = json.dumps(metadata[key], ensure_ascii=False, sort_keys=True) if isinstance(metadata[key], (dict, list)) else str(metadata[key])
                if key in result and result[key] != value:
                    raise ValueError(f"{path}: {key} differs from evaluation_metadata.json")
                result[key] = value
    return result


def paired_comparison(
    baseline: list[dict[str, Any]], corrupted: list[dict[str, Any]],
    baseline_provenance: dict[str, str] | None = None,
    corrupted_provenance: dict[str, str] | None = None,
    *, bootstrap_samples: int = 2000, seed: int = 42,
) -> dict[str, Any]:
    """Validate aligned labels and summarize paired recognition changes."""
    if not baseline or not corrupted:
        raise ValueError("Paired prediction sets must be nonempty")
    original = {row["id"]: row for row in baseline}
    changed = {row["id"]: row for row in corrupted}
    if len(original) != len(baseline) or len(changed) != len(corrupted):
        raise ValueError("Paired prediction sets contain duplicate IDs")
    if set(original) != set(changed):
        missing = sorted(set(original) - set(changed))[:5]
        extra = sorted(set(changed) - set(original))[:5]
        raise ValueError(f"Paired ID sets differ: missing={missing}, extra={extra}")
    for sample_id, row in original.items():
        if row["label"] != changed[sample_id]["label"]:
            raise ValueError(f"Paired label differs for id {sample_id!r}")
    base_meta, corrupt_meta = baseline_provenance or {}, corrupted_provenance or {}
    common = set(base_meta) & set(corrupt_meta)
    comparison_keys = common - {"checkpoint", "model_id"} if "checkpoint_sha256" in common else common
    for key in comparison_keys:
        if base_meta[key] != corrupt_meta[key]:
            raise ValueError(f"Paired {key} differs: {base_meta[key]!r} vs {corrupt_meta[key]!r}")
    condition_values = {row.get("condition", "") for row in corrupted}
    severity_values = {str(row.get("severity", "")) for row in corrupted}
    if len(condition_values) != 1 or len(severity_values) != 1:
        raise ValueError("A paired CSV must contain one condition and one severity")
    if next(iter(condition_values)) not in CORRUPTION_KINDS or next(iter(severity_values)) not in ("1", "2", "3"):
        raise ValueError("Paired corruptions require gaussian_blur/perspective/wave and severity 1/2/3")
    if any(row.get("condition") not in (None, "") or row.get("severity") not in (None, "") for row in baseline):
        if any(row.get("condition") != "clean" or str(row.get("severity")) != "0" for row in baseline):
            raise ValueError("Paired baseline must contain only clean condition and severity 0")
    counts = Counter()
    for sample_id, row in original.items():
        was_correct = row["label"] == row["prediction"]
        is_correct = changed[sample_id]["label"] == changed[sample_id]["prediction"]
        counts[(was_correct, is_correct)] += 1
    total = len(original)
    both_correct = counts[(True, True)]
    correct_to_wrong = counts[(True, False)]
    wrong_to_correct = counts[(False, True)]
    both_wrong = counts[(False, False)]
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    # A multinomial draw resamples aligned rows; it is equivalent to sampling
    # row indices but needs only four transition counts and little memory.
    probabilities = np.asarray([both_correct, correct_to_wrong, wrong_to_correct, both_wrong], dtype=float) / total
    draws = np.random.default_rng(seed).multinomial(total, probabilities, size=bootstrap_samples)
    deltas = (draws[:, 2] - draws[:, 1]) / total
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    baseline_metrics = summarize_predictions(baseline)
    corrupted_metrics = summarize_predictions(corrupted)
    baseline_cer, corrupted_cer = baseline_metrics["cer"], corrupted_metrics["cer"]
    # All conventions plus the actual checkpoint identity must be recorded to
    # claim verified provenance. Missing metadata is explicitly reported.
    verified = {"checkpoint_sha256", "decoder", "normalization", "preprocessing", "batch_padding"} <= common
    return {
        "condition": next(iter(condition_values)),
        "severity": next(iter(severity_values)),
        "num_samples": total,
        "baseline_word_accuracy": (both_correct + correct_to_wrong) / total,
        "corrupted_word_accuracy": (both_correct + wrong_to_correct) / total,
        "baseline_cer": baseline_cer,
        "corrupted_cer": corrupted_cer,
        "delta_cer": corrupted_cer - baseline_cer if baseline_cer is not None and corrupted_cer is not None else None,
        "delta_word_accuracy": (wrong_to_correct - correct_to_wrong) / total,
        "delta_ci_low": float(ci_low), "delta_ci_high": float(ci_high),
        "both_correct": both_correct, "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct, "both_wrong": both_wrong,
        "same_model_convention_verified": verified,
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
    }


def calibration_thresholds(manifest: Path) -> tuple[float, float, int]:
    from PIL import Image
    from .data import blur_score

    scores = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_path" not in (reader.fieldnames or []):
            raise ValueError(f"{manifest}: calibration requires image_path")
        for number, row in enumerate(reader, 2):
            if row.get("official_split") not in (None, "", "train"):
                raise ValueError(f"{manifest}:{number}: calibration must use training images, not {row['official_split']!r}")
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = manifest.parent / image_path
            try:
                with Image.open(image_path) as image:
                    scores.append(float(blur_score(image)))
            except (OSError, ValueError) as error:
                raise ValueError(f"{manifest}:{number}: cannot score {image_path}: {error}") from error
    if not scores:
        raise ValueError(f"{manifest}: empty calibration manifest")
    if not all(math.isfinite(score) and score >= 0 for score in scores):
        raise ValueError("Calibration returned invalid blur scores")
    low, high = np.quantile(scores, [1 / 3, 2 / 3])
    return float(low), float(high), len(scores)


def read_annotations(path: Path, predictions: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    return read_annotation_rows(path, {row["id"] for row in predictions})


def manual_blur_analysis(
    predictions: list[dict[str, Any]], annotations: dict[str, dict[str, str]],
    proxy_analysis: dict[str, Any],
) -> dict[str, Any]:
    tagged = [row for row in predictions if annotations.get(row["id"], {}).get("blur_manual")]
    groups = []
    for tag in BLUR_TAGS:
        selected = [row for row in tagged if annotations[row["id"]]["blur_manual"] == tag]
        group = group_summary(tag, selected)
        scores = [row["blur_score"] for row in selected if row.get("blur_score") is not None]
        group.update({"num_blur_scores": len(scores),
                      "median_blur_score": float(np.median(scores)) if scores else None,
                      "blur_score_q25": float(np.quantile(scores, 0.25)) if scores else None,
                      "blur_score_q75": float(np.quantile(scores, 0.75)) if scores else None})
        groups.append(group)
    result = {"enabled": bool(tagged), "num_annotations": len(tagged),
              "num_unannotated": len(predictions) - len(tagged),
              "coverage_fraction": len(tagged) / len(predictions), "groups": groups}
    validation: dict[str, Any] = {"enabled": False, "num_compared": 0}
    if tagged and proxy_analysis.get("enabled"):
        low, high = proxy_analysis["thresholds"]
        matrix = np.zeros((3, 3), dtype=int)
        severities, scores = [], []
        for row in tagged:
            score = row.get("blur_score")
            if score is None:
                continue
            manual_index = BLUR_TAGS.index(annotations[row["id"]]["blur_manual"])
            proxy_index = 0 if score <= low else 1 if score <= high else 2
            matrix[manual_index, proxy_index] += 1
            severities.append(manual_index)
            scores.append(score)
        count = int(matrix.sum())
        if count:
            agreement = float(np.trace(matrix[:, ::-1])) / count
            chance = float(np.dot(matrix.sum(axis=1), matrix.sum(axis=0)[::-1])) / (count * count)
            correlation = None
            if len(set(severities)) > 1 and len(set(scores)) > 1:
                from scipy.stats import rankdata
                correlation = float(np.corrcoef(rankdata(severities), rankdata(scores))[0, 1])
            validation = {
                "enabled": True, "num_compared": count,
                "num_missing_scores": len(tagged) - count,
                "manual_groups": list(BLUR_TAGS), "proxy_groups": list(BLUR_GROUPS),
                "confusion_matrix": matrix.tolist(),
                "proxy_to_manual_mapping": {"low_sharpness": "severe", "medium_sharpness": "moderate", "high_sharpness": "clear"},
                "agreement_fraction": agreement,
                "cohen_kappa_descriptive": (agreement - chance) / (1 - chance) if chance < 1 else None,
                "spearman_severity_vs_sharpness": correlation,
                "interpretation": "Descriptive validation against visual tags, not blur detector accuracy or inter-rater reliability. Train quantile bins are arbitrary categories, and annotation coverage/source matters.",
            }
    result["proxy_validation"] = validation
    return result


def annotation_provenance(path: Path | None, annotations: dict[str, dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "source_counts_by_field": {}}
    field_provenance = {}
    if path is not None:
        sidecar = annotation_provenance_path(path)
        if sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Annotation provenance must be a JSON object")
            expected = metadata.get("annotations_sha256")
            if expected != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Annotation provenance hash differs from the annotation CSV")
            field_provenance = metadata.get("field_provenance", {})
            if not isinstance(field_provenance, dict):
                raise ValueError("Malformed field_provenance")
            result.update({"available": True, "file": str(sidecar), "annotations_sha256": expected})
    for field in ("blur_manual", "distortion"):
        counts = Counter()
        for identifier, row in annotations.items():
            if row[field]:
                source = field_provenance.get(identifier, {}).get(field, {}).get("source", "unknown_existing")
                counts[source] += 1
        result["source_counts_by_field"][field] = dict(counts)
    return result


def _plot_proxy_validation(path: Path, validation: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(validation["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    display = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(3), BLUR_GROUPS, rotation=12)
    ax.set_yticks(range(3), BLUR_TAGS)
    ax.set_xlabel("Train-calibrated Laplacian proxy")
    ax.set_ylabel("Visual blur tag")
    ax.set_title(f'Visual tags vs. sharpness proxy (n={validation["num_compared"]})')
    for y in range(3):
        for x in range(3):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center",
                    color="white" if matrix[y, x] > matrix.max() / 2 else "black")
    fig.colorbar(display, ax=ax, label="Image count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_groups(path: Path, groups: list[dict[str, Any]], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(groups))
    values = np.asarray([group["word_accuracy"] or 0.0 for group in groups]) * 100
    low = np.asarray([group["word_accuracy_ci_low"] or 0.0 for group in groups]) * 100
    high = np.asarray([group["word_accuracy_ci_high"] or 0.0 for group in groups]) * 100
    errors = np.maximum(0, np.stack([values - low, high - values]))
    ax.bar(x, values, color="#245a8d", yerr=errors, capsize=5)
    ax.set_xticks(x, [f'{group["group"]}\nn={group["num_samples"]}' for group in groups])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Word accuracy (%)")
    ax.set_title(title + " (95% Wilson intervals)")
    ax.grid(axis="y", alpha=0.2)
    for index, group in enumerate(groups):
        if not group["num_samples"]:
            ax.text(index, 3, "N/A", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _points(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:+.2f}"


def _group_table(groups: list[dict[str, Any]]) -> str:
    lines = ["| Nhóm | n | Word accuracy | Khoảng 95% | CER micro |",
             "|---|---:|---:|---:|---:|"]
    for group in groups:
        interval = f'{_percentage(group["word_accuracy_ci_low"])} – {_percentage(group["word_accuracy_ci_high"])}'
        lines.append(f'| {group["group"]} | {group["num_samples"]} | {_percentage(group["word_accuracy"])} | {interval} | {_percentage(group["cer"])} |')
    return "\n".join(lines)


def _report(summary: dict[str, Any], errors: list[dict[str, Any]]) -> str:
    overall, blur = summary["overall"], summary["blur_analysis"]
    lines = ["# Phân tích lỗi CRNN trên IIIT5K", "",
             f'Số mẫu: **{overall["num_samples"]}**. Word accuracy: **{_percentage(overall["word_accuracy"])}** (khoảng Wilson 95%: {_percentage(overall["word_accuracy_ci_low"])} – {_percentage(overall["word_accuracy_ci_high"])}); CER micro: **{_percentage(overall["cer"])}**.', "",
             f'CER = {overall["character_edits"]} phép sửa / {overall["num_characters"]} ký tự nhãn chuẩn. So khớp dựa trên label và prediction đã xuất; script không tự đổi chữ hoa/thường hoặc xóa dấu câu.', "",
             "## Theo độ dài nhãn chuẩn", "", _group_table(summary["length_groups"]), "",
             "![Word accuracy theo độ dài](length_groups.png)", "",
             "## Theo độ mờ", ""]
    if blur["enabled"]:
        lines += [f'Ngưỡng thấp/cao: **{blur["thresholds"][0]:.6g} / {blur["thresholds"][1]:.6g}**, nguồn: `{blur["source"]}`. Điểm nhỏ biểu thị ít chi tiết tần số cao hơn; điểm này chỉ là đại diện cho độ sắc nét.', "",
                  "Nhóm low: score ≤ ngưỡng thấp; medium: ngưỡng thấp < score ≤ ngưỡng cao; high: score > ngưỡng cao. Mỗi ảnh được chuẩn hóa về chiều cao 64 trước khi tính phương sai Laplacian.", "",
                  _group_table(blur["groups"]), "", "![Word accuracy theo độ sắc nét](blur_groups.png)", ""]
        if blur["missing_scores"]:
            lines += [f'Có {blur["missing_scores"]} ảnh thiếu blur_score, đã loại khỏi phân nhóm độ mờ.', ""]
    else:
        lines += ["Chưa phân nhóm: cần --calibration từ tập train hoặc --blur-thresholds LOW HIGH. Không lấy ngưỡng từ lỗi trên tập test.", ""]
    manual = summary["manual_blur_analysis"]
    lines += ["## Theo nhãn độ mờ từ quan sát ảnh", "",
              f'Có {manual["num_annotations"]}/{overall["num_samples"]} ảnh có blur_manual ({_percentage(manual["coverage_fraction"])}); {manual["num_unannotated"]} ảnh chưa gán nhãn. Ô trống không được coi là clear.', ""]
    if manual["enabled"]:
        lines += [_group_table(manual["groups"]), "", "![Word accuracy theo nhãn mờ](manual_blur_groups.png)", ""]
        validation = manual["proxy_validation"]
        if validation["enabled"]:
            lines += [f'Đối chiếu {validation["num_compared"]} ảnh có cả nhãn quan sát và điểm proxy. Ánh xạ: low_sharpness→severe, medium_sharpness→moderate, high_sharpness→clear. Tỷ lệ trùng nhóm: {_percentage(validation["agreement_fraction"])}; Cohen kappa mô tả: {validation["cohen_kappa_descriptive"]}; Spearman giữa mức mờ (clear=0, moderate=1, severe=2) và điểm sắc nét: {validation["spearman_severity_vs_sharpness"]}.', "",
                      "![Đối chiếu nhãn mờ và proxy](manual_blur_proxy_agreement.png)", "",
                      "Đây là kiểm tra tính phù hợp của proxy trên phần ảnh đã quan sát. Ngưỡng chia ba từ train không phải ranh giới mờ chuẩn; kappa này không đo độ đồng thuận giữa nhiều người gán nhãn. Không suy ra chất lượng toàn bộ tập nếu mẫu gán nhãn được chọn theo lỗi mô hình.", ""]
    else:
        lines += ["Chưa có nhãn clear/moderate/severe. Dùng công cụ gán nhãn ngoại tuyến để quan sát ảnh gốc; không suy ra nhãn mờ tự động từ Laplacian.", ""]
    sources = summary["annotation_provenance"]["source_counts_by_field"]
    lines += [f'Nguồn nhãn được ghi trong provenance: blur_manual `{sources.get("blur_manual", {})}`; distortion `{sources.get("distortion", {})}`. unknown_existing nghĩa là chưa biết nguồn từ metadata.', ""]
    if any(counts.get("ai_review", 0) for counts in sources.values()):
        lines += ["Nhãn ai_review là đánh giá thăm dò bằng AI, chưa thay thế xác nhận của người gán nhãn. Tách kết luận từ nhãn này khỏi đánh giá đã được người kiểm tra.", ""]
    lines += ["## Theo biến dạng chữ tự nhiên", ""]
    if summary["distortion_groups"]:
        lines += [_group_table(summary["distortion_groups"]), "",
                  "![Word accuracy theo nhãn biến dạng](distortion_groups.png)", "",
                  f'Có {summary["num_distortion_annotations"]} ảnh được gán loại biến dạng, {overall["num_samples"] - summary["num_distortion_annotations"]} ảnh chưa có nhãn. Tỷ lệ khung hình không được dùng để suy ra biến dạng.', ""]
    else:
        lines += ["Chưa có nhãn biến dạng hợp lệ. Điền distortion trong mẫu annotation bằng none, perspective, curved, stylized hoặc mixed sau khi xem ảnh.", ""]
    lines += ["## Thử nghiệm độ bền trên các cặp ảnh", ""]
    if summary["paired_comparisons"]:
        lines += ["Delta = accuracy ảnh bị biến đổi − accuracy ảnh gốc, tính trên cùng ID và nhãn, đơn vị điểm phần trăm (pp). Khoảng delta là bootstrap theo cặp (2.000 lần, seed 42).", "",
                  "| Biến đổi | Mức | n | Gốc | Biến đổi | Delta (pp) | Khoảng delta 95% (pp) | Đúng→Sai | Sai→Đúng |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary["paired_comparisons"]:
            lines.append(f'| {row["condition"]} | {row["severity"]} | {row["num_samples"]} | {_percentage(row["baseline_word_accuracy"])} | {_percentage(row["corrupted_word_accuracy"])} | {_points(row["delta_word_accuracy"])} | {_points(row["delta_ci_low"])} – {_points(row["delta_ci_high"])} | {row["correct_to_wrong"]} | {row["wrong_to_correct"]} |')
        lines += ["", "CER micro dùng cùng tổng số ký tự nhãn chuẩn cho hai điều kiện. Delta CER = CER biến đổi − CER gốc; delta dương biểu thị nhiều lỗi ký tự hơn.", "",
                  "| Biến đổi | Mức | CER gốc | CER biến đổi | Delta CER (pp) |",
                  "|---|---:|---:|---:|---:|"]
        for row in summary["paired_comparisons"]:
            lines.append(f'| {row["condition"]} | {row["severity"]} | {_percentage(row["baseline_cer"])} | {_percentage(row["corrupted_cer"])} | {_points(row["delta_cer"])} |')
        if any(not row["same_model_convention_verified"] for row in summary["paired_comparisons"]):
            lines += ["", "Một số CSV thiếu checkpoint_sha256, decoder, normalization, preprocessing hoặc batch_padding. Các ID và nhãn đã khớp, nhưng chưa xác minh được rằng chúng dùng cùng mô hình và quy ước giải mã."]
        lines.append("")
    else:
        lines += ["Chưa cung cấp --paired-predictions. Các nhóm ảnh tự nhiên không thay thế thử nghiệm độ bền có đối chứng theo cặp.", ""]
    lines += ["## Ví dụ lỗi lớn", "", "Danh sách đầy đủ nằm trong errors.csv, sắp theo số phép sửa giảm dần.", "",
              "| ID | Nhãn | Dự đoán | Độ dài | Số phép sửa |", "|---|---|---|---:|---:|"]
    for row in errors[:15]:
        safe = lambda value: str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        lines.append(f'| {safe(row["id"])} | {safe(row["label"])} | {safe(row["prediction"])} | {row["length"]} | {row["edit_distance"]} |')
    if not errors:
        lines.append("| — | Không có lỗi | — | — | — |")
    lines += ["", "## Giới hạn diễn giải", ""]
    lines += [f"- {limitation}" for limitation in summary["limitations"]]
    return "\n".join(lines) + "\n"


def analyze(
    predictions: Path, output: Path, *, annotations: Path | None = None,
    calibration: Path | None = None, blur_thresholds: tuple[float, float] | None = None,
    paired_predictions: list[Path] | None = None,
) -> dict[str, Any]:
    rows = read_predictions(predictions)
    provenance = read_provenance(predictions, rows)
    length_groups = [group_summary(name, [row for row in rows if length_group(row["length"]) == name]) for name in LENGTH_GROUPS]
    blur: dict[str, Any] = {"enabled": False, "groups": [], "missing_scores": 0}
    if calibration is not None and blur_thresholds is not None:
        raise ValueError("Choose calibration or explicit blur thresholds, not both")
    if calibration is not None:
        low, high, count = calibration_thresholds(calibration)
        manifest_sha256 = hashlib.sha256(calibration.read_bytes()).hexdigest()
        split_metadata_path = calibration.parent / "split_metadata.json"
        if split_metadata_path.is_file():
            expected_hash = json.loads(split_metadata_path.read_text(encoding="utf-8")).get("manifest_sha256", {}).get("train.csv")
            if expected_hash and expected_hash != manifest_sha256:
                raise ValueError("Calibration manifest differs from the prepared train.csv; use the training split")
        source = f"train calibration: {calibration} (n={count}; SHA-256={manifest_sha256})"
    elif blur_thresholds is not None:
        low, high = blur_thresholds
        source = "explicit thresholds supplied by user"
    else:
        low = high = None
        source = "not configured"
    if low is not None and high is not None:
        if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high < low:
            raise ValueError("Blur thresholds must satisfy 0 <= LOW <= HIGH and be finite")
        buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in BLUR_GROUPS}
        missing_scores = 0
        for row in rows:
            score = row["blur_score"]
            if score is None:
                missing_scores += 1
            else:
                name = BLUR_GROUPS[0] if score <= low else BLUR_GROUPS[1] if score <= high else BLUR_GROUPS[2]
                buckets[name].append(row)
        blur = {"enabled": True, "thresholds": [low, high], "source": source,
                "missing_scores": missing_scores, "groups": [group_summary(name, buckets[name]) for name in BLUR_GROUPS]}
        if calibration is not None:
            blur["calibration_manifest"] = str(calibration)
            blur["calibration_manifest_sha256"] = manifest_sha256
            blur["num_calibration_samples"] = count
    annotation_rows = read_annotations(annotations, rows) if annotations else {}
    manual = manual_blur_analysis(rows, annotation_rows, blur)
    label_provenance = annotation_provenance(annotations, annotation_rows)
    tagged = [row for row in rows if annotation_rows.get(row["id"], {}).get("distortion")]
    distortion_groups = [group_summary(name, [row for row in tagged if annotation_rows[row["id"]]["distortion"] == name]) for name in DISTORTION_GROUPS] if tagged else []
    pairs = []
    for path in paired_predictions or []:
        changed = read_predictions(path)
        comparison = paired_comparison(rows, changed, provenance, read_provenance(path, changed))
        pairs.append({"predictions_file": str(path), **comparison})
    errors = sorted((row for row in rows if not row["correct"]), key=lambda row: (-row["edit_distance"], -row["length"], row["id"]))
    summary = {
        "predictions_file": str(predictions), "overall": group_summary("all", rows),
        "provenance": provenance, "length_groups": length_groups, "blur_analysis": blur,
        "manual_blur_analysis": manual, "annotation_provenance": label_provenance,
        "distortion_groups": distortion_groups, "num_distortion_annotations": len(tagged),
        "paired_comparisons": pairs,
        "limitations": [
            "Các nhóm độ dài, độ mờ và biến dạng trên ảnh tự nhiên là phân tầng quan sát. Độ dài, font, kích thước và nền có thể cùng thay đổi; chênh lệch accuracy không chứng minh quan hệ nhân quả.",
            "Phương sai Laplacian phụ thuộc font, tương phản, nhiễu, nền và phép đổi kích thước. Điểm này không phải nhãn mờ chuẩn; nên kiểm tra trực quan và bổ sung blur_manual.",
            "Nhãn quan sát clear/moderate/severe và biến dạng cần quy ước nhất quán, nguồn nhãn minh bạch và mẫu chọn độc lập với dự đoán. Phần ảnh chưa gán nhãn không thuộc nhóm none hoặc clear.",
            "Khoảng Wilson mô tả bất định của word accuracy nếu xem các ảnh là mẫu độc lập. Ảnh trùng hoặc cùng nguồn làm giả định độc lập yếu hơn; CER micro chưa có khoảng tin cậy trong báo cáo này.",
            "Nhóm ít mẫu có khoảng tin cậy rộng. Giữ tập test cho đánh giá cuối; chọn siêu tham số, epoch và ngưỡng phân tích bằng tập train/validation.",
            "Biến dạng tổng hợp có tham số biết trước nhưng không bao phủ toàn bộ biến dạng tự nhiên. Chỉ so sánh theo cặp khi cùng checkpoint, bộ giải mã, tiền xử lý và quy tắc nhãn.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output / "length_groups.csv", length_groups)
    _plot_groups(output / "length_groups.png", length_groups, "Ground-truth word length")
    if blur["enabled"]:
        _write_csv(output / "blur_groups.csv", blur["groups"])
        _plot_groups(output / "blur_groups.png", blur["groups"], "Train-calibrated sharpness proxy")
    if distortion_groups:
        _write_csv(output / "distortion_groups.csv", distortion_groups)
        _plot_groups(output / "distortion_groups.png", distortion_groups, "Manually annotated natural distortion")
    if manual["enabled"]:
        _write_csv(output / "manual_blur_groups.csv", manual["groups"])
        _plot_groups(output / "manual_blur_groups.png", manual["groups"], "Visually annotated natural blur")
        validation = manual["proxy_validation"]
        if validation["enabled"]:
            matrix = validation["confusion_matrix"]
            agreement_rows = [{"manual_blur": name, **dict(zip(BLUR_GROUPS, matrix[index])), "total": sum(matrix[index])} for index, name in enumerate(BLUR_TAGS)]
            _write_csv(output / "manual_blur_proxy_agreement.csv", agreement_rows)
            _plot_proxy_validation(output / "manual_blur_proxy_agreement.png", validation)
    if pairs:
        _write_csv(output / "paired_robustness.csv", pairs)
    error_fields = list(dict.fromkeys(key for row in rows for key in row))
    _write_csv(output / "errors.csv", errors, error_fields)
    (output / "report.md").write_text(_report(summary, errors), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích lỗi CRNN: độ dài, độ mờ và biến dạng chữ")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    blur = parser.add_mutually_exclusive_group()
    blur.add_argument("--calibration", type=Path, help="Manifest train dùng để cố định ngưỡng độ mờ")
    blur.add_argument("--blur-thresholds", type=float, nargs=2, metavar=("LOW", "HIGH"))
    parser.add_argument("--paired-predictions", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    try:
        summary = analyze(args.predictions, args.output, annotations=args.annotations,
                          calibration=args.calibration, blur_thresholds=args.blur_thresholds,
                          paired_predictions=args.paired_predictions)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f'Analyzed {summary["overall"]["num_samples"]} images. Report: {args.output / "report.md"}')


if __name__ == "__main__":
    main()
