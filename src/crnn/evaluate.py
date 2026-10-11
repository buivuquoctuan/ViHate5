"""Evaluate a saved CRNN and export per-image predictions for error analysis."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from .codec import Alphabet
from .data import BATCH_PADDING_POLICY, WordDataset, blur_score, collate_words
from .metrics import edit_distance, summarize_predictions
from .model import CRNN
from .train import manifest_digest, resolve_device


def load_checkpoint(path: Path, device: torch.device) -> tuple[CRNN, Alphabet, dict[str, Any]]:
    # Only tensors and primitive metadata are saved; disallow arbitrary pickle code.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format")
    if checkpoint.get("batch_padding") != BATCH_PADDING_POLICY:
        raise ValueError("Checkpoint batch padding does not match the current CNN padding policy")
    alphabet = Alphabet(checkpoint["alphabet"])
    config = checkpoint["model_config"]
    if config["num_classes"] != alphabet.num_classes:
        raise ValueError("Checkpoint alphabet does not match its output classes")
    model = CRNN(**config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, alphabet, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline/evaluation"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--corruption", choices=["gaussian_blur", "perspective", "wave"])
    parser.add_argument("--severity", type=int, choices=[1, 2, 3], default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative")
    if any((args.output / name).exists() for name in
           ("predictions.csv", "metrics.json", "evaluation_metadata.json")):
        raise FileExistsError("Evaluation output already exists. Choose a new --output directory.")
    device = resolve_device(args.device)
    model, alphabet, checkpoint = load_checkpoint(args.checkpoint, device)
    transform = None
    if args.corruption:
        from .distortions import apply_corruption
        transform = partial(apply_corruption, kind=args.corruption, severity=args.severity)
    dataset = WordDataset(args.manifest, transform=transform, **checkpoint["preprocessing"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_words, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    condition = args.corruption or "clean"
    severity = args.severity if args.corruption else 0
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            log_probs, lengths = model(batch["images"].to(device), batch["widths"])
            if not torch.isfinite(log_probs).all():
                raise FloatingPointError(f"Non-finite model outputs for samples {batch['ids']}")
            predictions = alphabet.decode_greedy(log_probs, lengths)
            for identifier, image_path, label, prediction in zip(
                    batch["ids"], batch["image_paths"], batch["labels"], predictions):
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                if transform is not None:
                    image = transform(image)
                rows.append({"id": identifier, "image_path": image_path, "label": label,
                             "prediction": prediction, "correct": int(prediction == label),
                             "edit_distance": edit_distance(label, prediction),
                             "length": len(label), "width": image.width, "height": image.height,
                             "blur_score": blur_score(image), "condition": condition,
                             "severity": severity})
    if not rows:
        raise ValueError("Cannot evaluate an empty manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in
           ("predictions.csv", "metrics.json", "evaluation_metadata.json")):
        raise FileExistsError("Evaluation output already exists. Choose a new --output directory.")
    columns = ["id", "image_path", "label", "prediction", "correct", "edit_distance",
               "length", "width", "height", "blur_score", "condition", "severity"]
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": manifest_digest(args.checkpoint),
                "checkpoint_epoch": checkpoint["epoch"],
                "manifest": str(args.manifest.resolve()),
                "manifest_sha256": manifest_digest(args.manifest),
                "condition": condition, "severity": severity,
                "decoder": "greedy_ctc", "normalization": "lowercase_alnum_strict",
                "device": str(device), "preprocessing": checkpoint["preprocessing"],
                "batch_padding": checkpoint["batch_padding"],
                "alphabet": alphabet.characters,
                "blur_measurement": "grayscale Laplacian variance at height 64"}
    if args.corruption:
        from .distortions import CORRUPTION_SPEC
        metadata["corruption_spec"] = CORRUPTION_SPEC
    metrics = summarize_predictions(rows)
    metrics["evaluation"] = metadata
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2),
                                                          encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Predictions: {(args.output / 'predictions.csv').resolve()}")


if __name__ == "__main__":
    main()
