"""Manifest-backed variable-width input with aspect-preserving preprocessing."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .codec import Alphabet

BATCH_PADDING_POLICY = "cnn_right_white_margin_24_v1"
CNN_RIGHT_MARGIN = 24


def read_manifest(manifest) -> list[dict]:
    manifest = Path(manifest).resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest}: CSV requires image_path,label.")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")
    seen = set()
    for index, row in enumerate(rows):
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        row["image_path"] = str(image_path)
        row["id"] = row.get("id") or str(index)
        if row["id"] in seen:
            raise ValueError(f"Duplicate sample ID {row['id']} in {manifest}")
        seen.add(row["id"])
    return rows


def preprocess_image(image: Image.Image, image_height=32, min_width=64, max_width=320):
    if image_height != 32 or not 4 <= min_width <= max_width or max_width % 4:
        raise ValueError("Use height32, 4<=min_width<=max_width, and max_width divisible by4.")
    gray = ImageOps.exif_transpose(image).convert("L")
    scale = min(image_height / gray.height, max_width / gray.width)
    width = max(1, min(max_width, round(gray.width * scale)))
    height = max(1, min(image_height, round(gray.height * scale)))
    resized = gray.resize((width, height), Image.Resampling.BICUBIC)
    canvas_width = min(max_width, 4 * math.ceil(max(min_width, width) / 4))
    canvas = Image.new("L", (canvas_width, image_height), 255)
    canvas.paste(resized, (0, (image_height - height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def blur_score(image: Image.Image) -> float:
    """Variance of discrete Laplacian at height64, a sharpness proxy, not blur truth."""
    gray = ImageOps.exif_transpose(image).convert("L")
    width = max(3, round(gray.width * 64 / gray.height))
    pixels = np.asarray(gray.resize((width, 64), Image.Resampling.BICUBIC), dtype=np.float64)
    center = pixels[1:-1, 1:-1]
    laplacian = (
        pixels[:-2, 1:-1] + pixels[2:, 1:-1]
        + pixels[1:-1, :-2] + pixels[1:-1, 2:] - 4 * center
    )
    return float(laplacian.var())


class WordDataset(Dataset):
    def __init__(self, manifest, image_height=32, min_width=64, max_width=320, transform=None):
        self.rows = read_manifest(manifest)
        self.image_height = image_height
        self.min_width = min_width
        self.max_width = max_width
        self.transform = transform
        alphabet = Alphabet()
        for row in self.rows:
            row["label"] = row["label"].lower()
            alphabet.encode(row["label"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        tensor = preprocess_image(image, self.image_height, self.min_width, self.max_width)
        return {
            "image": tensor, "width": tensor.shape[-1], "label": row["label"],
            "id": row["id"], "image_path": row["image_path"],
        }


def collate_words(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    # CNN receptive width is42. An additional24 pixels of white context keeps
    # valid columns away from the batch boundary, where implicit zero-padding
    # would otherwise differ between inference alone and in a larger batch.
    # These extra columns are excluded from packed BiLSTM and CTC lengths.
    width = max(sample["width"] for sample in samples) + CNN_RIGHT_MARGIN
    images = torch.ones(len(samples), 1, 32, width, dtype=torch.float32)
    for index, sample in enumerate(samples):
        images[index, :, :, :sample["width"]] = sample["image"]
    return {
        "images": images,
        "widths": torch.tensor([sample["width"] for sample in samples], dtype=torch.long),
        "labels": [sample["label"] for sample in samples],
        "ids": [sample["id"] for sample in samples],
        "image_paths": [sample["image_path"] for sample in samples],
    }
