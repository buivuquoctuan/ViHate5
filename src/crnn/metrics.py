"""Recognition metrics computed from strings, without reusing exported flags."""

from collections.abc import Mapping, Sequence
from typing import Any


def edit_distance(a: str, b: str) -> int:
    """Return unit-cost Levenshtein distance, using O(min(len(a), len(b))) memory."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("edit_distance expects two strings")
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def summarize_predictions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute exact word accuracy and micro CER for nonempty prediction rows.

    CER = sum(edit_distance(label, prediction)) / sum(len(label)). CER can exceed
    one, and is undefined (None) if every ground-truth string is empty. The mean
    normalized edit distance uses max(len(label), len(prediction), 1) per word.
    No case folding, stripping, or punctuation filtering is performed here.
    """
    if not rows:
        raise ValueError("Cannot summarize an empty prediction set")
    correct = edits = characters = 0
    normalized = 0.0
    for row in rows:
        label, prediction = row["label"], row["prediction"]
        distance = edit_distance(label, prediction)
        correct += label == prediction
        edits += distance
        characters += len(label)
        normalized += distance / max(len(label), len(prediction), 1)
    count = len(rows)
    return {
        "num_samples": count,
        "num_correct": correct,
        "num_errors": count - correct,
        "word_accuracy": correct / count,
        "cer": edits / characters if characters else None,
        "character_edits": edits,
        "num_characters": characters,
        "mean_normalized_edit_distance": normalized / count,
    }
