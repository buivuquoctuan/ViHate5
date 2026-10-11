"""Fixtures for analysis definitions, paired validity, and canvas preservation."""

import csv
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from crnn.analyze import (
    analyze,
    calibration_thresholds,
    group_summary,
    length_group,
    manual_blur_analysis,
    paired_comparison,
    read_annotations,
    read_predictions,
    wilson_interval,
)
from crnn.distortions import CORRUPTION_KINDS, apply_corruption
from crnn.metrics import edit_distance, summarize_predictions


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0), ("", "abc", 3), ("book", "bok", 1),
    ("aaa", "aa", 1), ("kitten", "sitting", 3), ("A", "a", 1),
])
def test_levenshtein_including_repeated_characters(a, b, expected):
    assert edit_distance(a, b) == expected
    assert edit_distance(b, a) == expected


def test_cer_is_micro_and_exported_correct_flags_are_not_trusted():
    rows = [{"label": "cat", "prediction": "cut", "correct": True},
            {"label": "abcdefghij", "prediction": "abcdefghij", "correct": False}]
    result = summarize_predictions(rows)
    assert result["word_accuracy"] == 0.5
    assert result["cer"] == pytest.approx(1 / 13)
    assert result["character_edits"] == 1
    assert result["num_characters"] == 13


def test_cer_can_exceed_one_and_empty_denominator_is_undefined():
    assert summarize_predictions([{"label": "a", "prediction": "bbbb"}])["cer"] == 4
    assert summarize_predictions([{"label": "", "prediction": "x"}])["cer"] is None
    with pytest.raises(ValueError, match="empty"):
        summarize_predictions([])


@pytest.mark.parametrize("length,expected", [(1, "1-3"), (3, "1-3"), (4, "4-6"),
                                               (6, "4-6"), (7, "7-9"), (9, "7-9"),
                                               (10, ">=10"), (25, ">=10")])
def test_length_boundaries(length, expected):
    assert length_group(length) == expected
    with pytest.raises(ValueError):
        length_group(0)


def test_empty_groups_are_undefined_and_wilson_extremes_are_finite():
    group = group_summary("no samples", [])
    assert group["num_samples"] == 0
    assert group["word_accuracy"] is None
    assert group["word_accuracy_ci_low"] is None
    low, high = wilson_interval(0, 5)
    assert 0 <= low <= high < 1
    low, high = wilson_interval(5, 5)
    assert 0 < low <= high <= 1


def _paired_fixture():
    baseline = [{"id": "a", "label": "cat", "prediction": "cat"},
                {"id": "b", "label": "dog", "prediction": "dig"},
                {"id": "c", "label": "book", "prediction": "bok"},
                {"id": "d", "label": "zz", "prediction": "zz"}]
    corrupted = [{"id": "c", "label": "book", "prediction": "book"},
                 {"id": "b", "label": "dog", "prediction": "dig"},
                 {"id": "d", "label": "zz", "prediction": "zz"},
                 {"id": "a", "label": "cat", "prediction": "cut"}]
    return baseline, [{**row, "condition": "wave", "severity": "2"} for row in corrupted]


def test_paired_alignment_uses_id_and_counts_transitions():
    baseline, corrupted = _paired_fixture()
    metadata = {"checkpoint_sha256": "a" * 64, "decoder": "greedy", "normalization": "lower_ascii", "preprocessing": "height32", "batch_padding": "cnn_right_white_margin_24_v1"}
    result = paired_comparison(baseline, corrupted, metadata, metadata)
    assert result["both_correct"] == result["both_wrong"] == 1
    assert result["correct_to_wrong"] == result["wrong_to_correct"] == 1
    assert result["baseline_word_accuracy"] == result["corrupted_word_accuracy"] == 0.5
    assert result["delta_word_accuracy"] == 0
    assert result["delta_ci_low"] <= 0 <= result["delta_ci_high"]
    assert result["same_model_convention_verified"]
    assert result == paired_comparison(baseline, corrupted, metadata, metadata)
    assert paired_comparison(baseline, corrupted, {**metadata, "checkpoint": "first.pt"},
                             {**metadata, "checkpoint": "renamed.pt"})["same_model_convention_verified"]
    incomplete = {key: value for key, value in metadata.items() if key != "batch_padding"}
    assert not paired_comparison(baseline, corrupted, incomplete, incomplete)["same_model_convention_verified"]


def test_paired_cer_delta_uses_the_common_character_denominator():
    baseline, corrupted = _paired_fixture()
    corrupted = [{**row, "prediction": "cuttt"} if row["id"] == "a" else row for row in corrupted]
    result = paired_comparison(baseline, corrupted)
    assert result["baseline_cer"] == pytest.approx(2 / 12)
    assert result["corrupted_cer"] == pytest.approx(4 / 12)
    assert result["delta_cer"] == pytest.approx(2 / 12)
    assert result["delta_word_accuracy"] == 0


def test_paired_mismatches_are_rejected():
    baseline, corrupted = _paired_fixture()
    with pytest.raises(ValueError, match="ID sets"):
        paired_comparison(baseline, corrupted[:-1])
    with pytest.raises(ValueError, match="label differs"):
        paired_comparison(baseline, [{**row, "label": "wrong"} if row["id"] == "a" else row for row in corrupted])
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(baseline, corrupted + [corrupted[0]])
    with pytest.raises(ValueError, match="checkpoint differs"):
        paired_comparison(baseline, corrupted, {"checkpoint": "one.pt"}, {"checkpoint": "two.pt"})
    with pytest.raises(ValueError, match="checkpoint_sha256 differs"):
        paired_comparison(baseline, corrupted, {"checkpoint_sha256": "a" * 64}, {"checkpoint_sha256": "b" * 64})
    with pytest.raises(ValueError, match="preprocessing differs"):
        paired_comparison(baseline, corrupted, {"preprocessing": "height32"}, {"preprocessing": "height64"})
    with pytest.raises(ValueError, match="batch_padding differs"):
        paired_comparison(baseline, corrupted, {"batch_padding": "old"}, {"batch_padding": "cnn_right_white_margin_24_v1"})
    with pytest.raises(ValueError, match="one condition"):
        paired_comparison(baseline, [{**corrupted[0], "condition": "perspective"}] + corrupted[1:])
    with pytest.raises(ValueError, match="baseline"):
        paired_comparison([{**row, "condition": "wave", "severity": "2"} for row in baseline], corrupted)


def _write_predictions(path, rows):
    fields = ["id", "image_path", "label", "prediction", "correct", "edit_distance",
              "length", "width", "height", "blur_score", "condition", "severity"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_reader_recomputes_metrics_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "predictions.csv"
    row = {"id": "a", "label": "aaa", "prediction": "aa", "length": 100,
           "correct": True, "edit_distance": 0, "blur_score": 10}
    _write_predictions(path, [row])
    result = read_predictions(path)[0]
    assert result["length"] == 3
    assert result["edit_distance"] == 1
    assert result["correct"] is False
    _write_predictions(path, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        read_predictions(path)


def test_analysis_outputs_explicit_bins_and_manual_distortion(tmp_path):
    predictions = tmp_path / "predictions.csv"
    rows = [{"id": "a", "label": "cat", "prediction": "cut", "blur_score": 10},
            {"id": "b", "label": "aaaaaa", "prediction": "aaaaaa", "blur_score": 20},
            {"id": "c", "label": "abcdefghij", "prediction": "abcdefghij", "blur_score": 30}]
    _write_predictions(predictions, rows)
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("id,blur_manual,distortion,notes\na,severe,perspective,checked\nb,clear,none,checked\nc,,,\n", encoding="utf-8")
    output = tmp_path / "analysis"
    result = analyze(predictions, output, annotations=annotations, blur_thresholds=(10.0, 20.0))
    assert [group["num_samples"] for group in result["length_groups"]] == [1, 1, 0, 1]
    assert [group["num_samples"] for group in result["blur_analysis"]["groups"]] == [1, 1, 1]
    assert result["num_distortion_annotations"] == 2
    assert result["manual_blur_analysis"]["num_annotations"] == 2
    assert result["manual_blur_analysis"]["coverage_fraction"] == pytest.approx(2 / 3)
    assert result["manual_blur_analysis"]["proxy_validation"]["num_compared"] == 2
    assert result["overall"]["cer"] == pytest.approx(1 / 19)
    assert result["overall"]["word_accuracy_ci_low"] < 2 / 3 < result["overall"]["word_accuracy_ci_high"]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["overall"]["num_samples"] == 3
    for filename in ["length_groups.csv", "length_groups.png", "blur_groups.csv", "blur_groups.png",
                     "distortion_groups.csv", "distortion_groups.png", "manual_blur_groups.csv",
                     "manual_blur_groups.png", "manual_blur_proxy_agreement.csv", "manual_blur_proxy_agreement.png",
                     "errors.csv", "report.md"]:
        assert (output / filename).is_file()
    with (output / "errors.csv").open(encoding="utf-8", newline="") as handle:
        errors = list(csv.DictReader(handle))
    assert [row["id"] for row in errors] == ["a"]


def test_no_test_fitted_blur_bins_or_aspect_ratio_distortion(tmp_path):
    predictions = tmp_path / "predictions.csv"
    _write_predictions(predictions, [{"id": "a", "label": "cat", "prediction": "cat",
                                    "width": 400, "height": 20, "blur_score": 10}])
    result = analyze(predictions, tmp_path / "analysis")
    assert result["blur_analysis"]["enabled"] is False
    assert result["distortion_groups"] == []


def test_annotation_requires_visual_tag_vocabulary(tmp_path):
    annotations = tmp_path / "annotations.csv"
    annotations.write_text("id,blur_manual,distortion,notes\na,,wide,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid distortion"):
        read_annotations(annotations, [{"id": "a"}])
    annotations.write_text("id,blur_manual,distortion,notes\na,blurry,none,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid blur_manual"):
        read_annotations(annotations, [{"id": "a"}])


def test_manual_blur_coverage_and_descriptive_proxy_agreement():
    rows = [{"id": "a", "label": "aa", "prediction": "aa", "blur_score": 20},
            {"id": "b", "label": "bb", "prediction": "bc", "blur_score": 10},
            {"id": "c", "label": "cc", "prediction": "c", "blur_score": 0},
            {"id": "d", "label": "dd", "prediction": "dd", "blur_score": 30}]
    annotations = {identifier: {"blur_manual": tag} for identifier, tag in zip("abc", ("clear", "moderate", "severe"))}
    result = manual_blur_analysis(rows, annotations, {"enabled": True, "thresholds": [5, 15]})
    assert result["num_annotations"] == 3
    assert result["num_unannotated"] == 1
    assert result["coverage_fraction"] == 0.75
    assert [group["num_samples"] for group in result["groups"]] == [1, 1, 1]
    assert result["groups"][1]["cer"] == 0.5
    assert result["groups"][0]["word_accuracy_ci_low"] < 1
    validation = result["proxy_validation"]
    assert validation["confusion_matrix"] == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    assert validation["agreement_fraction"] == validation["cohen_kappa_descriptive"] == 1
    assert validation["spearman_severity_vs_sharpness"] == pytest.approx(-1)
    result = manual_blur_analysis(rows, annotations, {"enabled": False})
    assert not result["proxy_validation"]["enabled"]


def test_calibration_uses_manifest_relative_training_images(tmp_path, monkeypatch):
    import crnn.data

    monkeypatch.setattr(crnn.data, "blur_score", lambda image: float(image.getpixel((0, 0))))
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index, value in enumerate((0, 3, 6)):
        Image.new("L", (20, 10), value).save(image_dir / f"{index}.png")
    manifest = tmp_path / "train.csv"
    manifest.write_text("image_path,label,official_split\nimages/0.png,a,train\nimages/1.png,b,train\nimages/2.png,c,train\n", encoding="utf-8")
    low, high, count = calibration_thresholds(manifest)
    assert (low, high, count) == pytest.approx((2, 4, 3))
    manifest.write_text("image_path,label,official_split\nimages/0.png,a,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training images"):
        calibration_thresholds(manifest)


@pytest.mark.parametrize("kind", CORRUPTION_KINDS)
@pytest.mark.parametrize("severity", (1, 2, 3))
def test_corruptions_have_same_canvas_and_are_deterministic(kind, severity):
    image = Image.new("RGB", (120, 48), "white")
    draw = ImageDraw.Draw(image)
    for box, color in [((0, 0, 10, 10), "red"), ((109, 0, 119, 10), "green"),
                       ((0, 37, 10, 47), "blue"), ((109, 37, 119, 47), "yellow")]:
        draw.rectangle(box, fill=color)
    first = apply_corruption(image, kind, severity)
    assert first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(apply_corruption(image, kind, severity)))
    if kind in ("perspective", "wave"):
        # Four edge markers must all remain visible after the transformation.
        pixels = np.asarray(first)
        assert np.any((pixels[..., 0] > 200) & (pixels[..., 1] < 40) & (pixels[..., 2] < 40))
        assert np.any((pixels[..., 0] < 40) & (pixels[..., 1] > 90) & (pixels[..., 2] < 40))
        assert np.any((pixels[..., 0] < 40) & (pixels[..., 1] < 40) & (pixels[..., 2] > 200))
        assert np.any((pixels[..., 0] > 200) & (pixels[..., 1] > 200) & (pixels[..., 2] < 40))


def test_corruption_arguments_are_validated():
    image = Image.new("L", (10, 10), 255)
    with pytest.raises(ValueError):
        apply_corruption(image, "cropping", 1)
    with pytest.raises(ValueError):
        apply_corruption(image, "wave", 0)
    with pytest.raises(ValueError):
        apply_corruption(image, "wave", 1.0)
