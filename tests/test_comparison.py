"""Compare saved experiments without allowing test-based model selection."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
from PIL import Image
import pytest

from crnn.codec import DEFAULT_CHARACTERS
from crnn.data import BATCH_PADDING_POLICY


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_experiments.py"
_SPEC = importlib.util.spec_from_file_location("compare_experiments", _SCRIPT)
comparison = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(comparison)

CONVENTIONS = {
    "alphabet": DEFAULT_CHARACTERS,
    "preprocessing": {"image_height": 32, "min_width": 64, "max_width": 320},
    "batch_padding": BATCH_PADDING_POLICY,
    "decoder": "greedy_ctc",
    "normalization": "lowercase_alnum_strict",
}
LABELS = ("cat", "dog", "book", "zz")
PREDICTION_COLUMNS = [
    "id", "image_path", "label", "prediction", "correct", "edit_distance",
    "length", "width", "height", "blur_score", "condition", "severity",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def update_json(path, **changes):
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(changes)
    write_json(path, value)


def predictions(values=("cat", "dig", "bok", "zz"), reverse=False):
    rows = [
        {"id": f"test-{index}", "image_path": f"Test/{index}.png", "label": label,
         "prediction": predicted, "correct": 0, "edit_distance": 999, "length": 999,
         "width": 64, "height": 32, "blur_score": 10.0, "condition": "clean", "severity": 0}
        for index, (label, predicted) in enumerate(zip(LABELS, values))
    ]
    return list(reversed(rows)) if reverse else rows


def write_predictions(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def make_run(root, recipe, seed, *, values=("cat", "dig", "bok", "zz"),
             validation=(0.8, 1.0), history=None, name=None, reverse=False, source=True):
    path = root / (name or f"{recipe}_seed{seed}")
    evaluation = path / "evaluation"
    evaluation.mkdir(parents=True)
    improved = recipe == "improved"
    config = {
        **CONVENTIONS,
        "seed": seed,
        "model_config": {"num_classes": 37, "hidden_size": 8, "dropout": 0.2 if improved else 0.0},
        "arguments": {"seed": seed, "output": str(path), "epochs": 3, "batch_size": 2,
                      "hidden_size": 8, "dropout": 0.2 if improved else 0.0,
                      "augmentation": "light" if improved else "none",
                      "optimizer": "adamw" if improved else "adam",
                      "scheduler": "plateau" if improved else "none",
                      "init_checkpoint": f"different/source/{seed}.pt" if improved else None},
        "manifests": {"train": {"sha256": "train-sha"}, "val": {"sha256": "val-sha"}},
        "initialization": {"checkpoint_sha256": f"source-{seed}"} if improved else None,
    }
    write_json(path / "config.json", config)
    # Hash verification is required; comparison must not deserialize model weights.
    (path / "best.pt").write_bytes(f"arbitrary checkpoint bytes {recipe} {seed}".encode())
    accuracy, loss = validation
    history = history if history is not None else [
        (1, max(0, accuracy - 0.05), loss + 1), (2, accuracy, loss),
        (3, max(0, accuracy - 0.1), 0.01),
    ]
    with (path / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_word_accuracy", "val_loss", "is_best"])
        writer.writeheader()
        writer.writerows({"epoch": epoch, "val_word_accuracy": acc, "val_loss": val_loss, "is_best": 0}
                         for epoch, acc, val_loss in history)
    best_epoch = min(history, key=lambda row: (-row[1], row[2], row[0]))[0]
    metadata = {
        **CONVENTIONS, "condition": "clean", "severity": 0,
        "checkpoint": str(path / "best.pt"), "checkpoint_sha256": sha256(path / "best.pt"),
        "checkpoint_epoch": best_epoch, "manifest_sha256": "common-test-sha",
    }
    write_json(evaluation / "evaluation_metadata.json", metadata)
    write_predictions(evaluation / "predictions.csv", predictions(values, reverse=reverse))
    if source:
        write_json(path / "source_sha256.json", {
            "snapshot": "historical", "files": {"src/crnn/model.py": "a" * 64},
        })
    return path


@pytest.fixture
def experiment_runs(tmp_path):
    baseline = [
        make_run(tmp_path, "baseline", 11),
        make_run(tmp_path, "baseline", 22, values=("cat", "dog", "bok", "zz")),
    ]
    improved = [
        make_run(tmp_path, "improved", 11, values=("cut", "dig", "bok", "zz"), validation=(0.9, 1.0), reverse=True),
        make_run(tmp_path, "improved", 22, values=LABELS, validation=(0.8, 0.5)),
    ]
    write_json(tmp_path / "experiment_plan.json", {
        "frozen_at": "2026-10-07T14:00:00Z", "selection": "validation_only",
        "recipes": ["baseline", "improved"],
    })
    return baseline, improved


def test_load_run_recomputes_metrics_and_preserves_historical_provenance(tmp_path):
    path = make_run(tmp_path, "baseline", 42,
                    history=[(1, 0.8, 2.0), (2, 0.8, 1.0), (3, 0.7, 0.01), (4, 0.8, 1.0)])
    record, rows = comparison.load_run(path, "baseline")
    assert record["best_epoch"] == 2  # Accuracy, loss, then earliest tied epoch.
    assert record["best_validation_word_accuracy"] == 0.8
    assert record["best_validation_loss"] == 1.0
    assert record["samples"] == 4
    assert record["correct"] == 2
    assert record["word_accuracy"] == 0.5
    assert record["character_edits"] == 2
    assert record["num_characters"] == 12
    assert record["cer"] == pytest.approx(2 / 12)
    assert rows[0]["correct"] is True
    assert rows[0]["length"] == 3
    assert rows[0]["edit_distance"] == 0
    assert record["word_accuracy_ci95"][0] < 0.5 < record["word_accuracy_ci95"][1]
    assert record["checkpoint_sha256"] == sha256(path / "best.pt")
    for key, relative in [
        ("config_sha256", "config.json"), ("history_sha256", "history.csv"),
        ("predictions_sha256", "evaluation/predictions.csv"),
        ("evaluation_metadata_sha256", "evaluation/evaluation_metadata.json"),
        ("source_metadata_sha256", "source_sha256.json"),
    ]:
        assert record[key] == sha256(path / relative)
    assert record["source_sha256_metadata"]["files"]["src/crnn/model.py"] == "a" * 64


def test_comparison_selects_by_validation_and_uses_sample_standard_deviation(experiment_runs):
    baseline, improved = experiment_runs
    result = comparison.compare_runs(baseline, improved, bootstrap_samples=200, bootstrap_seed=7)
    assert {"runs", "recipe_summary", "paired_seed_comparisons", "selected_improved",
            "main_baseline_counterpart", "unmatched_seeds", "conventions", "test_manifest_sha256",
            "recipe_metadata", "warnings", "metric_policy"} <= result.keys()
    selected = result["selected_improved"]
    assert selected["seed"] == 11
    assert selected["word_accuracy"] == 0.25
    assert next(row for row in result["runs"] if row["recipe"] == "improved" and row["seed"] == 22)["word_accuracy"] == 1.0
    assert result["main_baseline_counterpart"]["seed"] == 11
    assert result["metric_policy"]["test_metrics_used_for_selection"] is False
    assert selected["selection_basis"] == comparison.SELECTION_BASIS
    base_summary = result["recipe_summary"]["baseline"]
    assert base_summary["n"] == 2
    assert base_summary["word_accuracy_mean"] == pytest.approx(0.625)
    assert base_summary["word_accuracy_std"] == pytest.approx(math.sqrt(0.03125))
    assert base_summary["cer_mean"] == pytest.approx(0.125)
    assert base_summary["cer_std"] == pytest.approx(math.sqrt(1 / 288))
    assert result["recipe_summary"]["improved"]["word_accuracy_mean"] == pytest.approx(0.625)
    assert result["recipe_summary"]["improved"]["word_accuracy_std"] == pytest.approx(math.sqrt(0.28125))
    assert [row["seed"] for row in result["paired_seed_comparisons"]] == [11, 22]
    assert all(row["samples"] == 4 for row in result["paired_seed_comparisons"])
    # Different trained models are valid comparisons; each hash is verified locally.
    assert all(row["baseline_checkpoint_sha256"] != row["improved_checkpoint_sha256"]
               for row in result["paired_seed_comparisons"])
    assert result["unmatched_seeds"] == {"baseline": [], "improved": []}
    plan = Path(baseline[0]).parent / "experiment_plan.json"
    assert result["recipe_metadata"]["path"] == str(plan.resolve())
    assert result["recipe_metadata"]["sha256"] == sha256(plan)
    assert result["recipe_metadata"]["content"]["selection"] == "validation_only"


def test_single_seed_has_no_sample_sd_and_selection_ties_use_seed(tmp_path):
    baseline = [make_run(tmp_path, "baseline", 11)]
    improved = [
        make_run(tmp_path, "improved", 33, validation=(0.9, 0.8)),
        make_run(tmp_path, "improved", 22, validation=(0.9, 0.7)),
        make_run(tmp_path, "improved", 11, validation=(0.9, 0.7)),
    ]
    result = comparison.compare_runs(baseline, improved, bootstrap_samples=50)
    assert result["selected_improved"]["seed"] == 11
    assert result["recipe_summary"]["baseline"]["word_accuracy_std"] is None
    assert result["recipe_summary"]["baseline"]["cer_std"] is None
    assert result["unmatched_seeds"]["improved"] == [22, 33]


def test_unmatched_runs_are_descriptive_and_have_no_forced_counterpart(tmp_path):
    baseline = [make_run(tmp_path, "baseline", 11), make_run(tmp_path, "baseline", 22)]
    improved = [make_run(tmp_path, "improved", 11), make_run(tmp_path, "improved", 33, validation=(0.9, 0.5))]
    result = comparison.compare_runs(baseline, improved, bootstrap_samples=50)
    assert [row["seed"] for row in result["paired_seed_comparisons"]] == [11]
    assert result["recipe_summary"]["baseline"]["n"] == result["recipe_summary"]["improved"]["n"] == 2
    assert result["unmatched_seeds"] == {"baseline": [22], "improved": [33]}
    assert result["selected_improved"]["seed"] == 33
    assert result["main_baseline_counterpart"] is None


def paired_rows():
    baseline = predictions() + [{"id": "test-4", "label": "ab", "prediction": "ac"}]
    improved = [
        {**row, "prediction": prediction}
        for row, prediction in zip(baseline, ("cut", "dog", "book", "zz", "ac"))
    ]
    return baseline, list(reversed(improved))


def test_paired_bootstrap_aligns_by_id_and_repeats_with_fixed_seed():
    baseline, improved = paired_rows()
    result = comparison.paired_models(baseline, improved, seed=11, bootstrap_samples=500, bootstrap_seed=9)
    assert result["transitions"] == {"both_correct": 1, "correct_to_wrong": 1,
                                      "wrong_to_correct": 2, "both_wrong": 1}
    assert result["samples"] == 5
    assert result["delta_accuracy"] == pytest.approx(0.2)
    assert result["baseline_cer"] == pytest.approx(3 / 14)
    assert result["improved_cer"] == pytest.approx(2 / 14)
    assert result["delta_cer"] == pytest.approx(-1 / 14)
    assert -1 <= result["ci95"][0] <= result["ci95"][1] <= 1
    assert result["image_bootstrap"] == {"samples": 500, "seed": 9, "unit": "image", "confidence": 0.95}
    assert result == comparison.paired_models(list(reversed(baseline)), list(reversed(improved)),
                                             seed=11, bootstrap_samples=500, bootstrap_seed=9)


def test_all_paired_words_improve_has_exact_bootstrap_interval():
    improved = predictions(LABELS)
    baseline = [{**row, "prediction": ""} for row in improved]
    result = comparison.paired_models(baseline, improved, seed=42, bootstrap_samples=30)
    assert result["delta_accuracy"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
    assert result["transitions"]["wrong_to_correct"] == len(improved)


@pytest.mark.parametrize("mismatch,match", [("id", "ID sets"), ("label", "ground truth"), ("duplicate", "duplicate")])
def test_common_test_rejects_id_ground_truth_or_duplicate_mismatch(mismatch, match):
    baseline, improved = paired_rows()
    if mismatch == "id":
        improved = improved[:-1]
    elif mismatch == "label":
        improved = [{**row, "label": "changed"} if row["id"] == "test-0" else row for row in improved]
    else:
        improved.append(improved[0])
    with pytest.raises(ValueError, match=match):
        comparison.validate_common_test(baseline, improved)
    with pytest.raises(ValueError, match=match):
        comparison.paired_models(baseline, improved, seed=11, bootstrap_samples=10)


@pytest.mark.parametrize("key", comparison.CONVENTION_KEYS)
def test_missing_evaluation_convention_is_rejected(tmp_path, key):
    path = make_run(tmp_path, "baseline", 11)
    metadata_path = path / "evaluation/evaluation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop(key)
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match=f"requires {key}"):
        comparison.load_run(path, "baseline")


def changed_convention(key):
    return {"alphabet": DEFAULT_CHARACTERS[::-1],
            "preprocessing": {"image_height": 32, "min_width": 32, "max_width": 320},
            "batch_padding": "legacy_padding", "decoder": "beam_ctc",
            "normalization": "case_sensitive"}[key]


@pytest.mark.parametrize("key", comparison.CONVENTION_KEYS)
def test_config_and_evaluation_convention_mismatch_is_rejected(tmp_path, key):
    path = make_run(tmp_path, "baseline", 11)
    update_json(path / "evaluation/evaluation_metadata.json", **{key: changed_convention(key)})
    with pytest.raises(ValueError, match=f"disagree on {key}"):
        comparison.load_run(path, "baseline")


@pytest.mark.parametrize("key", comparison.CONVENTION_KEYS)
def test_comparison_rejects_different_conventions_between_valid_runs(tmp_path, key):
    baseline = make_run(tmp_path, "baseline", 11)
    improved = make_run(tmp_path, "improved", 11)
    for relative in ("config.json", "evaluation/evaluation_metadata.json"):
        update_json(improved / relative, **{key: changed_convention(key)})
    # Each artifact pair is internally consistent, but the evaluations differ.
    comparison.load_run(improved, "improved")
    with pytest.raises(ValueError, match=f"convention {key} differs"):
        comparison.compare_runs([baseline], [improved], bootstrap_samples=10)


@pytest.mark.parametrize("artifact,key,value", [
    ("metadata", "condition", "wave"), ("metadata", "severity", 1),
    ("predictions", "condition", "gaussian_blur"), ("predictions", "severity", 1),
])
def test_only_clean_severity_zero_evaluation_is_accepted(tmp_path, artifact, key, value):
    path = make_run(tmp_path, "baseline", 11)
    if artifact == "metadata":
        update_json(path / "evaluation/evaluation_metadata.json", **{key: value})
    else:
        rows = predictions()
        rows[0][key] = value
        write_predictions(path / "evaluation/predictions.csv", rows)
    with pytest.raises(ValueError, match="clean"):
        comparison.load_run(path, "baseline")


def test_changed_checkpoint_bytes_and_wrong_selected_epoch_are_rejected(tmp_path):
    changed_checkpoint = make_run(tmp_path, "baseline", 11)
    (changed_checkpoint / "best.pt").write_bytes(b"changed after evaluation")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        comparison.load_run(changed_checkpoint, "baseline")
    wrong_epoch = make_run(tmp_path, "baseline", 22)
    update_json(wrong_epoch / "evaluation/evaluation_metadata.json", checkpoint_epoch=3)
    with pytest.raises(ValueError, match="checkpoint epoch.*validation"):
        comparison.load_run(wrong_epoch, "baseline")


@pytest.mark.parametrize("change,match", [("ids", "ID sets"), ("labels", "ground truth"), ("manifest", "manifest SHA256")])
def test_comparison_rejects_changed_test_set(tmp_path, change, match):
    baseline = make_run(tmp_path, "baseline", 11)
    improved = make_run(tmp_path, "improved", 11)
    if change == "manifest":
        update_json(improved / "evaluation/evaluation_metadata.json", manifest_sha256="different-test-sha")
    else:
        rows = predictions()
        rows[0]["id" if change == "ids" else "label"] = "changed"
        write_predictions(improved / "evaluation/predictions.csv", rows)
    with pytest.raises(ValueError, match=match):
        comparison.compare_runs([baseline], [improved], bootstrap_samples=10)


def test_duplicate_seed_and_mixed_training_recipes_are_rejected(tmp_path):
    baseline = make_run(tmp_path, "baseline", 11)
    duplicate = make_run(tmp_path, "baseline", 11, name="duplicate_seed")
    improved = make_run(tmp_path, "improved", 11)
    with pytest.raises(ValueError, match="Duplicate seed"):
        comparison.compare_runs([baseline, duplicate], [improved], bootstrap_samples=10)
    changed_recipe = make_run(tmp_path, "baseline", 22)
    config = json.loads((changed_recipe / "config.json").read_text(encoding="utf-8"))
    config["arguments"]["lr"] = 0.01
    write_json(changed_recipe / "config.json", config)
    with pytest.raises(ValueError, match="Mixed training recipes"):
        comparison.compare_runs([baseline, changed_recipe], [improved], bootstrap_samples=10)


def test_missing_provenance_is_recorded_and_multiple_plans_require_explicit_choice(tmp_path):
    baseline = make_run(tmp_path / "a", "baseline", 11, source=False)
    improved = make_run(tmp_path / "b", "improved", 11, source=False)
    result = comparison.compare_runs([baseline], [improved], bootstrap_samples=10)
    assert result["recipe_metadata"] is None
    assert all(row["source_sha256_metadata"] is None and row["source_metadata_sha256"] is None
               for row in result["runs"])
    assert any("source_sha256.json" in text for text in result["warnings"])
    for parent in (baseline.parent, improved.parent):
        write_json(parent / "experiment_plan.json", {"plan": str(parent)})
    with pytest.raises(ValueError, match="Multiple experiment plans"):
        comparison.compare_runs([baseline], [improved], bootstrap_samples=10)
    explicit = baseline.parent / "experiment_plan.json"
    result = comparison.compare_runs([baseline], [improved], bootstrap_samples=10, recipe_metadata=explicit)
    assert result["recipe_metadata"]["path"] == str(explicit.resolve())


def test_recorded_recipes_and_missing_seeds_are_checked_against_plan(experiment_runs):
    baseline, improved = experiment_runs
    plan_path = baseline[0].parent / "experiment_plan.json"
    plan = {"baseline": {"lr": 1e-3, "batch_size": 2},
            "finetuning": {"dropout": 0.2, "augmentation": "light"}, "seeds": [11, 22, 33]}
    write_json(plan_path, plan)
    result = comparison.compare_runs(baseline, improved, bootstrap_samples=20)
    validation = result["recipe_metadata"]["validation"]
    assert validation["validated_fields"]["baseline"] == ["batch_size", "lr"]
    assert validation["validated_fields"]["improved"] == ["augmentation", "dropout"]
    assert validation["missing_planned_seeds"] == {"baseline": [33], "improved": [33]}
    plan["finetuning"]["dropout"] = 0.5
    write_json(plan_path, plan)
    with pytest.raises(ValueError, match="experiment plan for dropout"):
        comparison.compare_runs(baseline, improved, bootstrap_samples=20)
    plan["finetuning"]["dropout"] = 0.2
    plan["seeds"] = [11]
    write_json(plan_path, plan)
    with pytest.raises(ValueError, match="seed was not listed"):
        comparison.compare_runs(baseline, improved, bootstrap_samples=20)


def test_shared_pretraining_plan_checks_initialization_hash_and_source_recipe(experiment_runs):
    baseline, improved = experiment_runs
    plan_path = baseline[0].parent / "experiment_plan.json"
    write_json(plan_path, {"pretraining_initializations": 1,
                           "pretraining": {"seed": 42, "lr": 1e-3, "dropout": 0.1}})
    source_config = {"seed": 42, "arguments": {"lr": 1e-3},
                     "model_config": {"num_classes": 37, "hidden_size": 8, "dropout": 0.1}}
    provenance = {"checkpoint_sha256": "shared-pretraining-sha", "source_config": source_config}
    for run in improved:
        update_json(run / "config.json", initialization=provenance)
    result = comparison.compare_runs(baseline, improved, bootstrap_samples=20)
    assert result["recipe_metadata"]["validation"]["shared_pretraining_checkpoint_sha256"] == "shared-pretraining-sha"
    update_json(improved[0] / "config.json", initialization={**provenance, "checkpoint_sha256": "different-source"})
    with pytest.raises(ValueError, match="one shared pretraining"):
        comparison.compare_runs(baseline, improved, bootstrap_samples=20)
    update_json(improved[0] / "config.json", initialization={
        **provenance, "source_config": {**source_config, "arguments": {"lr": 0.01}},
    })
    with pytest.raises(ValueError, match="Pretraining differs.*lr"):
        comparison.compare_runs(baseline, improved, bootstrap_samples=20)


@pytest.mark.parametrize("samples,seed", [(0, 42), (10, -1)])
def test_bootstrap_arguments_are_rejected_even_without_common_seeds(tmp_path, samples, seed):
    baseline = make_run(tmp_path, "baseline", 11)
    improved = make_run(tmp_path, "improved", 22)
    with pytest.raises(ValueError, match="Bootstrap"):
        comparison.compare_runs([baseline], [improved], bootstrap_samples=samples, bootstrap_seed=seed)


def test_cli_exports_readable_artifacts_and_requires_explicit_overwrite(experiment_runs, tmp_path):
    baseline, improved = experiment_runs
    output = tmp_path / "comparison"
    argv = ["--baseline", *map(str, baseline), "--improved", *map(str, improved),
            "--output", str(output), "--bootstrap-samples", "100", "--bootstrap-seed", "7"]
    comparison.main(argv)
    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert result["selected_improved"]["seed"] == 11
    assert result["paired_seed_comparisons"][0]["image_bootstrap"]["seed"] == 7
    with (output / "runs.csv").open(encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    assert len(run_rows) == 4
    assert {(row["recipe"], row["seed"]) for row in run_rows} == {
        ("baseline", "11"), ("baseline", "22"), ("improved", "11"), ("improved", "22"),
    }
    with (output / "paired_seeds.csv").open(encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert len(pair_rows) == 2
    assert sum(int(pair_rows[0][key]) for key in ("both_correct", "correct_to_wrong", "wrong_to_correct", "both_wrong")) == 4
    report = (output / "comparison.md").read_text(encoding="utf-8")
    assert "So sánh CRNN" in report
    assert "Chọn seed **11**" in report
    assert "Accuracy/CER test không tham gia chọn run" in report
    assert "comparison.png" in report
    assert "\ufffd" not in report
    with Image.open(output / "comparison.png") as plot:
        assert plot.width > 100 and plot.height > 100
        plot.verify()
    assert ET.parse(output / "comparison.svg").getroot().tag.endswith("svg")
    assert matplotlib.get_backend().lower() == "agg"
    before = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    with pytest.raises(FileExistsError, match="overwrite"):
        comparison.main(argv)
    assert before == {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "comparison.json", {"sentinel": "old contents"})
    comparison.main([*argv, "--overwrite"])
    overwritten = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert "sentinel" not in overwritten
    assert overwritten["selected_improved"]["seed"] == 11
