"""Compare saved clean-test CRNN experiments without selecting by test scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from crnn.analyze import read_predictions, wilson_interval
from crnn.metrics import summarize_predictions


CONVENTION_KEYS = ("alphabet", "preprocessing", "batch_padding", "decoder", "normalization")
SELECTION_BASIS = "highest_validation_word_accuracy_then_lowest_validation_loss_then_lowest_seed"
WARNING_TEXT = [
    "Trung bình và SD mẫu giữa các seed là mô tả; ba seed chưa đủ để kết luận chắc chắn về độ ổn định huấn luyện.",
    "Khoảng bootstrap ghép cặp lấy lại mẫu ảnh test trong từng cặp seed; không đo biến thiên giữa các lần huấn luyện.",
    "Các seed dùng cùng tập test, vì vậy không coi số mẫu đánh giá là số seed nhân số ảnh test.",
    "Công thức cải tiến được phát triển sau khi đã quan sát baseline; giữ kế hoạch đã đóng băng và không dùng test để chọn checkpoint hoặc seed chính.",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def recipe_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Ignore seed/output/initial weight identity while retaining training rules."""
    arguments = config.get("arguments", {})
    defaults = {"epochs": 50, "batch_size": 32, "lr": 1e-3, "weight_decay": 0.0,
                "hidden_size": 256, "dropout": 0.0, "augmentation": "none",
                "optimizer": "adam", "scheduler": "none", "early_stopping_patience": 0,
                "grad_clip": 5.0, "min_width": 64, "max_width": 320}
    signature = {key: arguments.get(key, default) for key, default in defaults.items()}
    signature["dropout"] = config.get("model_config", {}).get("dropout", signature["dropout"])
    signature["warm_started"] = bool(arguments.get("init_checkpoint") or config.get("initialization"))
    signature["train_manifest_sha256"] = config.get("manifests", {}).get("train", {}).get("sha256")
    signature["val_manifest_sha256"] = config.get("manifests", {}).get("val", {}).get("sha256")
    signature["model_num_classes"] = config.get("model_config", {}).get("num_classes")
    return signature


def load_run(path: Path | str, recipe: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    config = read_json(path / "config.json")
    metadata = read_json(path / "evaluation/evaluation_metadata.json")
    rows = read_predictions(path / "evaluation/predictions.csv")
    for key in CONVENTION_KEYS:
        if metadata.get(key) in (None, "", {}):
            raise ValueError(f"{path}: evaluation metadata requires {key}")
    if metadata.get("condition") != "clean" or str(metadata.get("severity")) != "0":
        raise ValueError(f"{path}: comparison requires clean test evaluation at severity 0")
    if any(row.get("condition") != "clean" or str(row.get("severity")) != "0" for row in rows):
        raise ValueError(f"{path}: predictions must all have clean condition and severity 0")
    for key in CONVENTION_KEYS:
        if key in config and config[key] != metadata[key]:
            raise ValueError(f"{path}: config and evaluation disagree on {key}")
    checkpoint_path = path / "best.pt"
    checkpoint_sha = file_sha256(checkpoint_path)
    if metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{path}: evaluation checkpoint SHA256 does not match best.pt")
    seed = config.get("seed", config.get("arguments", {}).get("seed"))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{path}: run seed must be an integer")
    with (path / "history.csv").open(encoding="utf-8-sig", newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError(f"{path}: empty training history")
    candidates = []
    epochs = set()
    for raw in history:
        epoch = int(raw["epoch"])
        accuracy, loss = float(raw["val_word_accuracy"]), float(raw["val_loss"])
        if epoch < 1 or epoch in epochs or not math.isfinite(loss) or not 0 <= accuracy <= 1:
            raise ValueError(f"{path}: invalid validation history")
        epochs.add(epoch)
        candidates.append((epoch, accuracy, loss))
    best_epoch, best_accuracy, best_loss = min(candidates, key=lambda item: (-item[1], item[2], item[0]))
    if metadata.get("checkpoint_epoch") != best_epoch:
        raise ValueError(f"{path}: checkpoint epoch does not follow validation accuracy/loss selection")
    metrics = summarize_predictions(rows)
    interval = wilson_interval(metrics["num_correct"], metrics["num_samples"])
    source_path = path / "source_sha256.json"
    source_metadata = read_json(source_path) if source_path.is_file() else None
    record = {
        "recipe": recipe, "seed": seed, "run_dir": str(path), "best_epoch": best_epoch,
        "best_validation_word_accuracy": best_accuracy, "best_validation_loss": best_loss,
        "samples": metrics["num_samples"], "correct": metrics["num_correct"],
        "word_accuracy": metrics["word_accuracy"], "cer": metrics["cer"],
        "character_edits": metrics["character_edits"], "num_characters": metrics["num_characters"],
        "word_accuracy_ci95": list(interval), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": str(checkpoint_path), "evaluation_metadata": metadata,
        "evaluation_metadata_sha256": file_sha256(path / "evaluation/evaluation_metadata.json"),
        "predictions_sha256": file_sha256(path / "evaluation/predictions.csv"),
        "config_sha256": file_sha256(path / "config.json"),
        "history_sha256": file_sha256(path / "history.csv"),
        "source_sha256_metadata": source_metadata,
        "source_metadata_sha256": file_sha256(source_path) if source_metadata is not None else None,
        "recipe_signature": recipe_signature(config), "config": config,
    }
    return record, rows


def validate_common_test(reference: list[dict], candidate: list[dict]) -> None:
    expected = {row["id"]: row["label"] for row in reference}
    actual = {row["id"]: row["label"] for row in candidate}
    if len(expected) != len(reference) or len(actual) != len(candidate):
        raise ValueError("Test predictions contain duplicate IDs")
    if expected.keys() != actual.keys():
        raise ValueError("Test ID sets differ between runs")
    for identifier in expected:
        if expected[identifier] != actual[identifier]:
            raise ValueError(f"Test ground truth differs for id {identifier!r}")


def paired_models(baseline: list[dict], improved: list[dict], *, seed: int,
                  bootstrap_samples: int = 2000, bootstrap_seed: int = 42) -> dict[str, Any]:
    validate_common_test(baseline, improved)
    if not baseline or bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("Paired bootstrap requires rows, positive samples, and a nonnegative seed")
    changed = {row["id"]: row for row in improved}
    counts = Counter((row["label"] == row["prediction"],
                      changed[row["id"]]["label"] == changed[row["id"]]["prediction"])
                     for row in baseline)
    transitions = {"both_correct": counts[True, True], "correct_to_wrong": counts[True, False],
                   "wrong_to_correct": counts[False, True], "both_wrong": counts[False, False]}
    n = len(baseline)
    probabilities = np.array(list(transitions.values()), dtype=float) / n
    draws = np.random.default_rng(bootstrap_seed).multinomial(n, probabilities, size=bootstrap_samples)
    deltas = (draws[:, 2] - draws[:, 1]) / n
    low, high = np.quantile(deltas, [0.025, 0.975])
    base_metrics, new_metrics = summarize_predictions(baseline), summarize_predictions(improved)
    return {"seed": seed, "samples": n,
            "baseline_word_accuracy": base_metrics["word_accuracy"],
            "improved_word_accuracy": new_metrics["word_accuracy"],
            "baseline_cer": base_metrics["cer"], "improved_cer": new_metrics["cer"],
            "delta_accuracy": new_metrics["word_accuracy"] - base_metrics["word_accuracy"],
            "delta_cer": new_metrics["cer"] - base_metrics["cer"],
            "ci95": [float(low), float(high)], "transitions": transitions,
            "image_bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed,
                                "unit": "image", "confidence": 0.95}}


def aggregate_recipe(records: list[dict]) -> dict[str, Any]:
    accuracy, cer = [row["word_accuracy"] for row in records], [row["cer"] for row in records]
    return {"n": len(records), "seeds": sorted(row["seed"] for row in records),
            "word_accuracy_mean": statistics.mean(accuracy),
            "word_accuracy_std": statistics.stdev(accuracy) if len(accuracy) > 1 else None,
            "cer_mean": statistics.mean(cer), "cer_std": statistics.stdev(cer) if len(cer) > 1 else None}


def validate_recipe_plan(plan: dict[str, Any], groups: dict[str, list[dict]]) -> dict[str, Any]:
    """Check recorded recipe fields when the plan uses the project schema."""
    result: dict[str, Any] = {"validated_fields": {}, "missing_planned_seeds": {}}
    for recipe, plan_key in (("baseline", "baseline"), ("improved", "finetuning")):
        expected = plan.get(plan_key)
        if isinstance(expected, dict):
            fields = []
            for key, value in expected.items():
                if key not in groups[recipe][0]["recipe_signature"]:
                    continue
                fields.append(key)
                if any(row["recipe_signature"][key] != value for row in groups[recipe]):
                    raise ValueError(f"{recipe}: recorded recipe differs from experiment plan for {key}")
            result["validated_fields"][recipe] = sorted(fields)
        if isinstance(plan.get("seeds"), list):
            actual = {row["seed"] for row in groups[recipe]}
            planned = set(plan["seeds"])
            if not actual <= planned:
                raise ValueError(f"{recipe}: seed was not listed in the experiment plan")
            result["missing_planned_seeds"][recipe] = sorted(planned - actual)
    if plan.get("pretraining_initializations") == 1:
        initializations = [row["config"].get("initialization") for row in groups["improved"]]
        if any(not isinstance(source, dict) or not source.get("checkpoint_sha256") for source in initializations):
            raise ValueError("Experiment plan requires recorded pretraining initialization for every improved run")
        if len({source["checkpoint_sha256"] for source in initializations}) != 1:
            raise ValueError("Experiment plan specifies one shared pretraining initialization")
        result["shared_pretraining_checkpoint_sha256"] = initializations[0]["checkpoint_sha256"]
        expected = plan.get("pretraining", {})
        for source in initializations:
            config = source.get("source_config")
            if not isinstance(config, dict):
                raise ValueError("Pretraining source config is missing from warm-start provenance")
            signature = recipe_signature(config)
            for key, value in expected.items():
                actual = config.get("seed") if key == "seed" else signature.get(key)
                if actual != value:
                    raise ValueError(f"Pretraining differs from experiment plan for {key}")
    return result


def compare_runs(baseline: list[Path], improved: list[Path], *, bootstrap_samples: int = 2000,
                 bootstrap_seed: int = 42, recipe_metadata: Path | None = None) -> dict[str, Any]:
    if not baseline or not improved:
        raise ValueError("Both baseline and improved recipes need at least one run")
    if bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("Bootstrap samples must be positive and bootstrap seed nonnegative")
    groups: dict[str, list[dict]] = {}
    predictions: dict[tuple[str, int], list[dict]] = {}
    records = []
    reference_rows = None
    conventions = None
    manifest_sha = None
    for recipe, paths in (("baseline", baseline), ("improved", improved)):
        group = []
        signature = None
        for path in paths:
            record, rows = load_run(path, recipe)
            run_key = recipe, record["seed"]
            if run_key in predictions:
                raise ValueError(f"Duplicate seed {record['seed']} within {recipe}")
            current_conventions = {key: record["evaluation_metadata"][key] for key in CONVENTION_KEYS}
            if reference_rows is None:
                reference_rows, conventions = rows, current_conventions
                manifest_sha = record["evaluation_metadata"].get("manifest_sha256")
            else:
                validate_common_test(reference_rows, rows)
                for convention in CONVENTION_KEYS:
                    if conventions[convention] != current_conventions[convention]:
                        raise ValueError(f"Evaluation convention {convention} differs between runs")
                candidate_sha = record["evaluation_metadata"].get("manifest_sha256")
                if manifest_sha is not None and candidate_sha is not None and candidate_sha != manifest_sha:
                    raise ValueError("Test manifest SHA256 differs between runs")
            if signature is None:
                signature = record["recipe_signature"]
            elif signature != record["recipe_signature"]:
                raise ValueError(f"Mixed training recipes within {recipe}; compare each recipe separately")
            predictions[run_key] = rows
            records.append(record)
            group.append(record)
        groups[recipe] = sorted(group, key=lambda row: row["seed"])
    common_seeds = sorted(set(row["seed"] for row in groups["baseline"]) &
                          set(row["seed"] for row in groups["improved"]))
    pairs = []
    for seed in common_seeds:
        pair = paired_models(predictions["baseline", seed], predictions["improved", seed], seed=seed,
                             bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
        pair["baseline_checkpoint_sha256"] = next(row["checkpoint_sha256"] for row in groups["baseline"] if row["seed"] == seed)
        pair["improved_checkpoint_sha256"] = next(row["checkpoint_sha256"] for row in groups["improved"] if row["seed"] == seed)
        pairs.append(pair)
    selected = min(groups["improved"], key=lambda row: (-row["best_validation_word_accuracy"],
                                                       row["best_validation_loss"], row["seed"]))
    selected = {**selected, "selection_basis": SELECTION_BASIS}
    counterpart = next((row for row in groups["baseline"] if row["seed"] == selected["seed"]), None)
    if recipe_metadata is None:
        candidates = {(Path(path).resolve().parent / "experiment_plan.json") for path in baseline + improved}
        found = [path for path in candidates if path.is_file()]
        if len(found) > 1:
            raise ValueError("Multiple experiment plans found; specify --recipe-metadata")
        recipe_metadata = found[0] if found else None
    plan = ({"path": str(recipe_metadata.resolve()), "sha256": file_sha256(recipe_metadata),
             "content": read_json(recipe_metadata)} if recipe_metadata is not None else None)
    if plan is not None:
        plan["validation"] = validate_recipe_plan(plan["content"], groups)
    warnings = list(WARNING_TEXT)
    if plan is None:
        warnings.append("Không tìm thấy metadata kế hoạch công thức đã đóng băng; không xác nhận thời điểm đăng ký trước.")
    elif plan["content"].get("pretraining_initializations") == 1:
        warnings.append("Các seed cải tiến dùng chung một checkpoint tiền huấn luyện; SD chỉ mô tả thay đổi của fine-tuning, không bao gồm biến thiên tiền huấn luyện.")
    if plan is not None and any(plan["validation"]["missing_planned_seeds"].values()):
        warnings.append("Chưa có đủ mọi seed trong kế hoạch; xem missing_planned_seeds trước khi coi đây là so sánh hoàn tất.")
    missing_sources = [row["run_dir"] for row in records if row["source_sha256_metadata"] is None]
    if missing_sources:
        warnings.append("Thiếu source_sha256.json ở một số run; xem trường source_sha256_metadata.")
    unmatched = {recipe: sorted(set(row["seed"] for row in groups[recipe]) - set(common_seeds))
                 for recipe in groups}
    if any(unmatched.values()):
        warnings.append("Một số seed không có cặp tương ứng; chúng chỉ tham gia thống kê mô tả công thức.")
    return {"runs": sorted(records, key=lambda row: (row["recipe"], row["seed"])),
            "recipe_summary": {recipe: aggregate_recipe(rows) for recipe, rows in groups.items()},
            "paired_seed_comparisons": pairs, "selected_improved": selected,
            "main_baseline_counterpart": counterpart, "unmatched_seeds": unmatched,
            "conventions": conventions, "test_manifest_sha256": manifest_sha,
            "recipe_metadata": plan, "warnings": warnings,
            "metric_policy": {"word_accuracy": "exact_string_match", "cer": "micro_Levenshtein_edits/ground_truth_characters",
                              "seed_std": "sample_standard_deviation_ddof_1",
                              "main_selection": SELECTION_BASIS, "test_metrics_used_for_selection": False}}


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# So sánh CRNN baseline và công thức cải tiến", "",
             "Các chỉ số được tính lại từ nhãn và dự đoán test, sau khi kiểm tra ID, nhãn thật, quy ước đánh giá và SHA256 checkpoint.", "",
             "## Từng lần chạy", "",
             "| Công thức | Seed | Epoch tốt nhất | Accuracy val | Loss val | n test | Đúng | Accuracy test | Wilson 95% | CER micro |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for row in result["runs"]:
        lines.append(f"| {row['recipe']} | {row['seed']} | {row['best_epoch']} | {pct(row['best_validation_word_accuracy'])} | "
                     f"{row['best_validation_loss']:.4f} | {row['samples']} | {row['correct']} | {pct(row['word_accuracy'])} | "
                     f"{pct(row['word_accuracy_ci95'][0])}–{pct(row['word_accuracy_ci95'][1])} | {pct(row['cer'])} |")
    lines += ["", "## Thống kê giữa các seed", "",
              "| Công thức | Số seed | Accuracy trung bình | SD mẫu accuracy | CER trung bình | SD mẫu CER |",
              "|---|---:|---:|---:|---:|---:|"]
    for recipe, row in result["recipe_summary"].items():
        lines.append(f"| {recipe} | {row['n']} | {pct(row['word_accuracy_mean'])} | {pct(row['word_accuracy_std'])} | "
                     f"{pct(row['cer_mean'])} | {pct(row['cer_std'])} |")
    lines += ["", "SD của chỉ số tỷ lệ được biểu diễn bằng điểm phần trăm. SD không phải khoảng tin cậy.", "",
              "## So sánh ghép cặp theo cùng seed", "",
              "Delta là improved trừ baseline; delta accuracy được biểu diễn bằng điểm phần trăm.", "",
              "| Seed | Delta accuracy | Bootstrap ảnh 95% | Delta CER | Đúng→sai | Sai→đúng | Cùng đúng | Cùng sai |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in result["paired_seed_comparisons"]:
        t = row["transitions"]
        lines.append(f"| {row['seed']} | {100 * row['delta_accuracy']:+.2f} pp | "
                     f"[{100 * row['ci95'][0]:+.2f}, {100 * row['ci95'][1]:+.2f}] pp | {100 * row['delta_cer']:+.2f} pp | "
                     f"{t['correct_to_wrong']} | {t['wrong_to_correct']} | {t['both_correct']} | {t['both_wrong']} |")
    selected = result["selected_improved"]
    lines += ["", "## Run chính được chọn bằng validation", "",
              f"Chọn seed **{selected['seed']}**, epoch **{selected['best_epoch']}** của `{selected['run_dir']}`: "
              f"Word Accuracy validation {pct(selected['best_validation_word_accuracy'])}, loss validation {selected['best_validation_loss']:.4f}.", "",
              "Quy tắc: accuracy validation cao nhất, rồi loss validation thấp nhất; nếu vẫn hòa chọn seed nhỏ nhất. "
              "Accuracy/CER test không tham gia chọn run. Baseline đối chiếu chính dùng cùng seed nếu có.", "",
              "## Giới hạn diễn giải", ""]
    lines += [f"- {warning}" for warning in result["warnings"]]
    lines += ["", "## Provenance", "",
              "`comparison.json` giữ metadata đánh giá, SHA256 checkpoint/CSV/config/history, snapshot hash mã nguồn của từng run "
              "và kế hoạch công thức. Hash mã nguồn được giữ theo snapshot lịch sử; không ép bằng mã nguồn hiện tại.", ""]
    lines += ["Hash rút gọn bên dưới chỉ dùng để nhận diện; hash đầy đủ nằm trong JSON và CSV.", "",
              "| Công thức | Seed | Run | Checkpoint SHA256 (12 ký tự đầu) | Source metadata SHA256 (12 ký tự đầu) |",
              "|---|---:|---|---|---|"]
    for row in result["runs"]:
        source_sha = row["source_metadata_sha256"]
        lines.append(f"| {row['recipe']} | {row['seed']} | `{row['run_dir']}` | "
                     f"`{row['checkpoint_sha256'][:12]}` | {f'`{source_sha[:12]}`' if source_sha else 'N/A'} |")
    lines += [""]
    plan = result["recipe_metadata"]
    if plan:
        lines += [f"Kế hoạch: `{plan['path']}`; SHA256 `{plan['sha256']}`.", ""]
    lines += ["![So sánh accuracy và CER giữa các seed](comparison.png)", ""]
    return "\n".join(lines)


def plot_comparison(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    palette = ["#2864a0", "#d97720"]
    for axis, metric, title in zip(axes, ("word_accuracy", "cer"),
                                    ("Word Accuracy trên test", "CER micro trên test")):
        for index, recipe in enumerate(("baseline", "improved")):
            rows = [row for row in result["runs"] if row["recipe"] == recipe]
            jitter = np.linspace(-0.08, 0.08, len(rows)) if len(rows) > 1 else [0]
            values = [100 * row[metric] for row in rows]
            axis.scatter(index + np.asarray(jitter), values, color=palette[index], s=38, zorder=3)
            for offset, value, row in zip(jitter, values, rows):
                axis.annotate(str(row["seed"]), (index + offset, value), xytext=(0, 7),
                              textcoords="offset points", ha="center", fontsize=8)
            summary = result["recipe_summary"][recipe]
            mean, std = 100 * summary[f"{metric}_mean"], summary[f"{metric}_std"]
            axis.errorbar(index, mean, yerr=100 * std if std is not None else 0,
                          fmt="_", markersize=24, elinewidth=2, capsize=6,
                          color=palette[index], zorder=2)
        axis.set_xticks([0, 1], ["Baseline", "Cải tiến"])
        axis.set_xlim(-0.4, 1.4)
        axis.set_ylim(bottom=0)
        axis.set_title(title)
        axis.set_ylabel("Tỷ lệ (%)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Từng seed và trung bình ± SD mẫu; số trên điểm là seed", fontsize=11)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--improved", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe-metadata", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = compare_runs(args.baseline, args.improved, bootstrap_samples=args.bootstrap_samples,
                          bootstrap_seed=args.bootstrap_seed, recipe_metadata=args.recipe_metadata)
    names = ("runs.csv", "paired_seeds.csv", "comparison.json", "comparison.md", "comparison.png", "comparison.svg")
    if not args.overwrite and any((args.output / name).exists() for name in names):
        raise FileExistsError("Comparison output already exists; choose a fresh directory or --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    columns = ["recipe", "seed", "run_dir", "best_epoch", "best_validation_word_accuracy",
               "best_validation_loss", "samples", "correct", "word_accuracy", "cer",
               "checkpoint_sha256", "checkpoint_path", "source_metadata_sha256", "predictions_sha256"]
    with (args.output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["runs"])
    columns = ["seed", "samples", "baseline_word_accuracy", "improved_word_accuracy", "delta_accuracy",
               "ci95_low", "ci95_high", "baseline_cer", "improved_cer", "delta_cer",
               "both_correct", "correct_to_wrong", "wrong_to_correct", "both_wrong"]
    with (args.output / "paired_seeds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, **row["transitions"], "ci95_low": row["ci95"][0], "ci95_high": row["ci95"][1]}
                         for row in result["paired_seed_comparisons"])
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "comparison.md").write_text(markdown_report(result), encoding="utf-8")
    plot_comparison(result, args.output)
    print(f"Comparison: {(args.output / 'comparison.md').resolve()}")


if __name__ == "__main__":
    main()
