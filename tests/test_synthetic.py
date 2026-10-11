import csv
import hashlib
import json
import random
from pathlib import Path

import pytest

from crnn.synthetic import render_word, training_vocabulary
from crnn.codec import minimum_ctc_steps
from crnn.data import preprocess_image


def test_synthetic_source_rejects_frozen_validation_manifest(tmp_path):
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train.write_text("label,official_split\nbook,train\n", encoding="utf-8")
    val.write_text("label,official_split\nshop,train\n", encoding="utf-8")
    (tmp_path / "split_metadata.json").write_text(json.dumps({"manifest_sha256": {
        "train.csv": hashlib.sha256(train.read_bytes()).hexdigest()}}), encoding="utf-8")
    assert training_vocabulary(train) == ["book"]
    with pytest.raises(ValueError, match="frozen train"):
        training_vocabulary(val)


def test_synthetic_source_strict_labels_and_split(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("label,official_split\nbook,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training rows"):
        training_vocabulary(source)
    source.write_text("label\ncafé\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        training_vocabulary(source)


def test_rendered_word_is_reproducible_and_ctc_feasible():
    font = Path("C:/Windows/Fonts/arial.ttf")
    if not font.exists():
        pytest.skip("Local Windows font absent")
    first = render_word("book123", font, random.Random(42))
    second = render_word("book123", font, random.Random(42))
    assert first.size == second.size
    assert first.tobytes() == second.tobytes()
    assert preprocess_image(first).shape[-1] // 4 >= minimum_ctc_steps("book123")
