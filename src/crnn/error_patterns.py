"""Summarize deterministic edit alignments and observable word difficulty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .codec import minimum_ctc_steps
from .metrics import edit_distance


def alignment(reference: str, prediction: str) -> list[dict]:
    """One minimum-cost alignment, with match/substitute/delete/insert tie order.

    Other optimal alignments can allocate ambiguous errors differently. Positions
    refer to original zero-based strings; an absent symbol is the empty string.
    """
    table = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        table[i][0] = i
    for j in range(len(prediction) + 1):
        table[0][j] = j
    for i, a in enumerate(reference, 1):
        for j, b in enumerate(prediction, 1):
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + (a != b))
    i, j = len(reference), len(prediction)
    operations = []
    while i or j:
        if i and j and reference[i - 1] == prediction[j - 1] and table[i][j] == table[i - 1][j - 1]:
            operation = {"operation": "match", "reference": reference[i - 1],
                         "prediction": prediction[j - 1], "reference_position": i - 1,
                         "prediction_position": j - 1}
            i, j = i - 1, j - 1
        elif i and j and table[i][j] == table[i - 1][j - 1] + 1:
            operation = {"operation": "substitution", "reference": reference[i - 1],
                         "prediction": prediction[j - 1], "reference_position": i - 1,
                         "prediction_position": j - 1}
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            operation = {"operation": "deletion", "reference": reference[i - 1],
                         "prediction": "", "reference_position": i - 1,
                         "prediction_position": j}
            i -= 1
        else:
            operation = {"operation": "insertion", "reference": "",
                         "prediction": prediction[j - 1], "reference_position": i,
                         "prediction_position": j - 1}
            j -= 1
        operations.append(operation)
    return list(reversed(operations))


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    chars = sum(len(row["label"]) for row in rows)
    correct = sum(row["label"] == row["prediction"] for row in rows)
    edits = sum(edit_distance(row["label"], row["prediction"]) for row in rows)
    return {"num_samples": count, "num_correct": correct,
            "word_accuracy": correct / count if count else None,
            "character_edits": edits, "num_characters": chars,
            "cer": edits / chars if chars else None}


def analyze(predictions: Path, output: Path, train_manifest: Path) -> dict:
    with predictions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with train_manifest.open(encoding="utf-8-sig", newline="") as handle:
        train_words = {row["label"] for row in csv.DictReader(handle)}
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Predictions must contain unique nonempty sample IDs.")
    metadata = json.loads(predictions.with_name("evaluation_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("condition") != "clean":
        raise ValueError("Pattern analysis expects original clean-image predictions.")
    preprocessing = metadata["preprocessing"]
    if isinstance(preprocessing, str):
        preprocessing = json.loads(preprocessing)
    output.mkdir(parents=True, exist_ok=True)
    confusion = Counter()
    totals = Counter()
    sample_rows = []
    groups = {name: [] for name in ["adjacent_repeat", "no_adjacent_repeat", "width_limited",
                                   "not_width_limited", "seen_word", "unseen_word"]}
    for row in rows:
        operations = alignment(row["label"], row["prediction"])
        counts = Counter(op["operation"] for op in operations)
        if counts["substitution"] + counts["deletion"] + counts["insertion"] != int(row["edit_distance"]):
            raise ValueError(f"Saved edit distance disagrees with strings: {row['id']}")
        totals.update(counts)
        confusion.update((op["operation"], op["reference"], op["prediction"])
                         for op in operations if op["operation"] != "match")
        repeated = any(a == b for a, b in zip(row["label"], row["label"][1:]))
        width_limited = (preprocessing["image_height"] * int(row["width"]) /
                         int(row["height"]) > preprocessing["max_width"])
        seen = row["label"] in train_words
        groups["adjacent_repeat" if repeated else "no_adjacent_repeat"].append(row)
        groups["width_limited" if width_limited else "not_width_limited"].append(row)
        groups["seen_word" if seen else "unseen_word"].append(row)
        sample_rows.append({"id": row["id"], "label": row["label"], "prediction": row["prediction"],
                            "substitutions": counts["substitution"], "deletions": counts["deletion"],
                            "insertions": counts["insertion"], "adjacent_repeat": int(repeated),
                            "minimum_ctc_steps": minimum_ctc_steps(row["label"]),
                            "width_limited": int(width_limited), "seen_word": int(seen)})
    confusion_rows = [{"operation": op, "reference": ref, "prediction": pred, "count": count}
                      for (op, ref, pred), count in sorted(confusion.items(), key=lambda item: (-item[1], item[0]))]
    for filename, values, fields in [
        ("sample_edit_patterns.csv", sample_rows, list(sample_rows[0])),
        ("character_confusions.csv", confusion_rows, ["operation", "reference", "prediction", "count"])]:
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    result = {"overall": summarize(rows), "operation_counts": dict(totals),
              "observable_groups": {key: summarize(value) for key, value in groups.items()},
              "top_character_errors": confusion_rows[:20],
              "provenance": {"predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                             "train_manifest_sha256": hashlib.sha256(train_manifest.read_bytes()).hexdigest(),
                             "checkpoint_sha256": metadata["checkpoint_sha256"]},
              "alignment_policy": "minimum Levenshtein cost; backward tie order match/substitute/delete/insert",
              "limitations": ["Ambiguous optimal alignments can change the allocation among edit types.",
                              "Observable groups have different label/length distributions; associations are descriptive.",
                              "Deletion errors do not establish CTC as the cause."]}
    (output / "error_patterns.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Error pattern analysis ready: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.predictions, args.output, args.train_manifest)


if __name__ == "__main__":
    main()
