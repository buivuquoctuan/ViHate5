"""Render reproducible synthetic word images using train-only text and local fonts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .codec import Alphabet, minimum_ctc_steps
from .data import preprocess_image
from .distortions import apply_corruption

FONT_FILES = (
    "arial.ttf", "arialbd.ttf", "arialbi.ttf", "ariali.ttf", "ariblk.ttf",
    "calibri.ttf", "calibrib.ttf", "calibrii.ttf", "calibriz.ttf",
    "cambria.ttc", "cambriab.ttf", "cambriai.ttf", "cambriaz.ttf",
    "consola.ttf", "consolab.ttf", "consolai.ttf", "cour.ttf", "courbd.ttf", "courbi.ttf",
    "georgia.ttf", "georgiab.ttf", "georgiai.ttf", "georgiaz.ttf",
    "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuisl.ttf",
    "times.ttf", "timesbd.ttf", "timesbi.ttf", "timesi.ttf", "impact.ttf",
    "comic.ttf", "comicbd.ttf", "comici.ttf", "framd.ttf", "framdit.ttf",
    "Candara.ttf", "Candarab.ttf", "Candarai.ttf", "corbel.ttf", "corbelb.ttf",
    "pala.ttf", "palab.ttf", "palai.ttf",
)
VALIDATION_FONT_FILES = ("verdana.ttf", "verdanab.ttf", "verdanai.ttf", "tahoma.ttf", "tahomabd.ttf")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_vocabulary(manifest: Path) -> list[str]:
    """Reject official val/test manifests rather than drawing labels from them."""
    metadata_path = manifest.parent / "split_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata.get("manifest_sha256", {}).get("train.csv")
        if expected and sha256(manifest) != expected:
            raise ValueError("Synthetic text source must match the frozen train.csv hash.")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "label" not in (reader.fieldnames or []):
            raise ValueError("Text-source manifest needs a label column.")
        rows = list(reader)
    if not rows:
        raise ValueError("Text-source manifest is empty.")
    alphabet = Alphabet()
    labels = set()
    for row in rows:
        if row.get("official_split") not in (None, "", "train"):
            raise ValueError("Only official training rows may supply synthetic text.")
        label = row["label"].lower()
        alphabet.encode(label)
        labels.add(label)
    return sorted(labels)


@lru_cache(maxsize=256)
def load_font(path: str, size: int):
    return ImageFont.truetype(path, size=size)


def random_word(rng: random.Random) -> str:
    length = rng.choices(range(1, 13), weights=[1, 4, 8, 12, 14, 14, 12, 10, 8, 6, 4, 3])[0]
    if rng.random() < 0.18:
        alphabet = "0123456789"
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        if rng.random() < 0.15:
            alphabet += "0123456789"
    return "".join(rng.choice(alphabet) for _ in range(length))


def render_word(label: str, font_path: Path, rng: random.Random) -> Image.Image:
    font = load_font(str(font_path), rng.choice([36, 40, 44, 48, 52, 56, 60]))
    roll = rng.random()
    text = label.upper() if roll < 0.5 else label.title() if roll < 0.65 else label
    stroke = rng.choices([0, 1, 2], weights=[8, 2, 1])[0]
    box = font.getbbox(text, stroke_width=stroke)
    margin_x, margin_y = rng.randint(4, 12), rng.randint(4, 10)
    width = max(8, box[2] - box[0] + 2 * margin_x)
    height = max(8, box[3] - box[1] + 2 * margin_y)
    dark = rng.random() < 0.35
    if dark:
        background = tuple(rng.randint(5, 70) for _ in range(3))
        foreground = tuple(rng.randint(185, 255) for _ in range(3))
    else:
        background = tuple(rng.randint(185, 255) for _ in range(3))
        foreground = tuple(rng.randint(0, 70) for _ in range(3))
    image = Image.new("RGB", (width, height), background)
    noise_seed = rng.randrange(2**32)
    noise_rng = np.random.default_rng(noise_seed)
    if rng.random() < 0.6:
        array = np.asarray(image, dtype=np.float32)
        gradient = np.linspace(-rng.uniform(0, 20), rng.uniform(0, 20), width)[None, :, None]
        noise = noise_rng.normal(0, rng.uniform(0.5, 7), (height, width, 1))
        image = Image.fromarray(np.clip(array + gradient + noise, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    if rng.random() < 0.2:
        for _ in range(rng.randint(1, 4)):
            colour = tuple(max(0, min(255, channel + rng.randint(-25, 25))) for channel in background)
            draw.line((rng.randint(0, width - 1), 0, rng.randint(0, width - 1), height - 1), fill=colour, width=1)
    position = (margin_x - box[0], margin_y - box[1])
    stroke_colour = foreground if rng.random() < 0.8 else background
    draw.text(position, text, font=font, fill=foreground, stroke_width=stroke, stroke_fill=stroke_colour)
    if rng.random() < 0.35:
        image = image.rotate(rng.uniform(-3, 3), resample=Image.Resampling.BICUBIC,
                             expand=True, fillcolor=background)
    if rng.random() < 0.12:
        image = apply_corruption(image, "perspective", 1)
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.8)))
    return image


def write_rows(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "label", "text_source", "font", "split"])
        writer.writeheader()
        writer.writerows(rows)


def generate(source: Path, output: Path, train_samples=20000, val_samples=1000, seed=20261007,
             font_directory: Path = Path("C:/Windows/Fonts")) -> dict:
    if train_samples < 1 or val_samples < 1:
        raise ValueError("Both synthetic splits need positive sample counts.")
    source = source.resolve()
    vocabulary = training_vocabulary(source)
    train_fonts = [font_directory / name for name in FONT_FILES if (font_directory / name).is_file()]
    val_fonts = [font_directory / name for name in VALIDATION_FONT_FILES if (font_directory / name).is_file()]
    if not train_fonts or not val_fonts:
        raise FileNotFoundError("Need local Latin training fonts and held-out Verdana/Tahoma fonts; pass --fonts.")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("Synthetic output already exists; choose a fresh directory.")
    output.mkdir(parents=True)
    rng = random.Random(seed)
    train_labels = set(vocabulary)
    generated_labels = {"train": set(), "val": set()}
    source_counts = Counter()
    row_sets = {}
    for split, count, fonts in [("train", train_samples, train_fonts), ("val", val_samples, val_fonts)]:
        image_directory = output / split
        image_directory.mkdir()
        rows = []
        for index in range(count):
            text_source = "iiit5k_train_label" if split == "train" and rng.random() < 0.4 else "random_alphanumeric"
            label = rng.choice(vocabulary) if text_source == "iiit5k_train_label" else random_word(rng)
            if split == "val":
                # Only synthetic-train vocabulary is consulted, never IIIT val/test labels.
                while label in train_labels:
                    label = random_word(rng)
            else:
                train_labels.add(label)
            font = rng.choice(fonts)
            image = render_word(label, font, rng)
            steps = preprocess_image(image).shape[-1] // 4
            if steps < minimum_ctc_steps(label):
                raise ValueError(f"Synthetic CTC infeasible: {label!r} in font {font.name}")
            filename = image_directory / f"{index:06d}.png"
            image.save(filename, compress_level=2)
            rows.append({"id": f"synthetic/{split}/{index:06d}",
                         "image_path": Path(os.path.relpath(filename, output)).as_posix(),
                         "label": label, "text_source": text_source, "font": font.name, "split": split})
            generated_labels[split].add(label)
            source_counts[f"{split}:{text_source}"] += 1
            if (index + 1) % 2000 == 0:
                print(f"Rendered {split}: {index + 1}/{count}", flush=True)
        path = output / f"{split}.csv"
        write_rows(path, rows)
        row_sets[split] = path
    metadata = {
        "generator_version": "local-font-random-words-v1", "seed": seed,
        "counts": {"train": train_samples, "val": val_samples},
        "source_manifest": str(source), "source_manifest_sha256": sha256(source),
        "text_sources": dict(source_counts), "source_vocabulary_size": len(vocabulary),
        "iiit5k_val_or_test_used": False, "benchmark_lexicon_used": False,
        "synthetic_word_overlap_train_val": len(generated_labels['train'] & generated_labels['val']),
        "validation_policy": "novel random labels and held-out Verdana/Tahoma font files",
        "fonts": {str(font): sha256(font) for font in train_fonts + val_fonts},
        "train_fonts": [font.name for font in train_fonts], "val_fonts": [font.name for font in val_fonts],
        "manifest_sha256": {path.name: sha256(path) for path in row_sets.values()},
        "preprocessing": {"image_height": 32, "min_width": 64, "max_width": 320},
        "reference": "https://arxiv.org/abs/1406.2227",
        "limitation": "Small procedural dataset; not Synth90k and not a reproduction of the original text engine.",
    }
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Synthetic manifests ready: {output}", flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/manifests/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_v1"))
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--fonts", type=Path, default=Path("C:/Windows/Fonts"))
    args = parser.parse_args(argv)
    generate(args.source, args.output, args.train_samples, args.val_samples, args.seed, args.fonts)


if __name__ == "__main__":
    main()
