import csv
import io
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.io import savemat

from crnn.codec import Alphabet, minimum_ctc_steps
from crnn.data import WordDataset, blur_score, collate_words, preprocess_image
from crnn.download import safe_extract
from crnn.prepare import grouped_split, load_official_rows


def test_ctc_blank_order_retains_double_letter_and_masks_padding():
    alphabet = Alphabet()
    b, o, k = [alphabet.encode(char)[0] for char in "bok"]
    path = [b, b, o, 0, o, k, k, b]
    logits = torch.full((len(path), 1, alphabet.num_classes), -10.0)
    for index, value in enumerate(path):
        logits[index, 0, value] = 10
    assert alphabet.decode_greedy(logits, [7]) == ["book"]
    assert minimum_ctc_steps("book") == 5
    with pytest.raises(ValueError, match="Unsupported"):
        alphabet.encode("café")


def test_preprocess_fits_without_stretching_and_pads_white():
    narrow = Image.new("L", (8, 32), 0)
    image = preprocess_image(narrow)
    assert image.shape == (1, 32, 64)
    assert torch.all(image[:, :, :8] == -1)
    assert torch.all(image[:, :, 8:] == 1)
    wide = preprocess_image(Image.new("L", (640, 32), 0))
    assert wide.shape == (1, 32, 320)
    assert torch.all(wide[:, :8, :] == 1)
    assert torch.all(wide[:, 8:24, :] == -1)


def test_matlab_parser_and_grouped_split(tmp_path):
    root = tmp_path / "IIIT5K"
    (root / "train").mkdir(parents=True)
    records = np.empty((1, 12), dtype=[("ImgName", "O"), ("GroundTruth", "O")])
    for index in range(12):
        path = f"train/{index}.png"
        Image.new("RGB", (48, 20), (index, 80, 90)).save(root / path)
        records[0, index] = (path, f"WORD{index // 2}")
    savemat(root / "traindata.mat", {"trainData": records})
    rows, _ = load_official_rows(root.resolve(), "train")
    assert len(rows) == 12
    assert rows[0]["label"] == "word0"
    train, val = grouped_split(rows, val_fraction=0.25, seed=42)
    assert len(train) + len(val) == 12
    assert len(val) == 2  # Target3 is unattainable with paired word groups.
    assert not {r["label"] for r in train} & {r["label"] for r in val}
    assert grouped_split(rows, 0.25, 42) == (train, val)


def test_manifest_paths_and_collate(tmp_path):
    for name, width in [("a.png", 80), ("b.png", 120)]:
        Image.new("L", (width, 32), 0).save(tmp_path / name)
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "label"])
        writer.writeheader()
        writer.writerows([{"id": "a", "image_path": "a.png", "label": "BOOK"},
                          {"id": "b", "image_path": "b.png", "label": "shop"}])
    data = WordDataset(path)
    batch = collate_words([data[0], data[1]])
    assert batch["images"].shape == (2, 1, 32, 144)
    assert batch["widths"].tolist() == [80, 120]
    assert batch["labels"] == ["book", "shop"]
    assert torch.all(batch["images"][0, :, :, 80:] == 1)
    assert blur_score(Image.new("L", (100, 30), 128)) == 0


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("safe/link", "symlink")])
def test_archive_rejects_escape_and_links(tmp_path, name, kind):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            archive.addfile(info)
        else:
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    output = tmp_path / "raw"
    output.mkdir()
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(path, output)
    assert not (tmp_path / "escape").exists()
