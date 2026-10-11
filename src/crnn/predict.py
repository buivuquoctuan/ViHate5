"""Recognize one or more cropped word images with a saved CRNN."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data import collate_words, preprocess_image
from .evaluate import load_checkpoint
from .train import resolve_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON predictions file.")
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    model, alphabet, checkpoint = load_checkpoint(args.checkpoint, device)
    samples = []
    for index, path in enumerate(args.image):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = preprocess_image(image, **checkpoint["preprocessing"])
        samples.append({"image": tensor, "width": tensor.shape[-1], "label": "",
                        "id": str(index), "image_path": str(path.resolve())})
    batch = collate_words(samples)
    with torch.inference_mode():
        log_probs, lengths = model(batch["images"].to(device), batch["widths"])
        predictions = alphabet.decode_greedy(log_probs, lengths)
    rows = [{"image_path": sample["image_path"], "prediction": prediction}
            for sample, prediction in zip(samples, predictions)]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
