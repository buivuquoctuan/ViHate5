"""Exercise CLI integration using tiny synthetic fixtures, not IIIT5K results."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def run(*arguments):
    subprocess.run([sys.executable, "-m", *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError("Choose a fresh smoke --output directory.")
    root.mkdir(parents=True)
    for split, words in [("train", ["book", "shop", "123", "cafe"]), ("val", ["moon", "abc"])]:
        with (root / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "label"])
            writer.writeheader()
            for index, word in enumerate(words):
                image = Image.new("RGB", (80 + index * 12, 32), "white")
                ImageDraw.Draw(image).text((4, 8), word, fill="black")
                filename = f"{split}_{index}.png"
                image.save(root / filename)
                writer.writerow({"id": f"{split}_{index}", "image_path": filename, "label": word})
    run("crnn.train", "--train", root / "train.csv", "--val", root / "val.csv",
        "--output", root / "model", "--epochs", 1, "--batch-size", 2,
        "--hidden-size", 16, "--device", args.device)
    checkpoint = root / "model/best.pt"
    for condition in ["clean", "wave"]:
        arguments = ["crnn.evaluate", "--checkpoint", checkpoint, "--manifest", root / "val.csv",
                     "--output", root / condition, "--batch-size", 2, "--device", args.device]
        if condition != "clean":
            arguments += ["--corruption", condition, "--severity", 2]
        run(*arguments)
    run("crnn.analyze", "--predictions", root / "clean/predictions.csv", "--output", root / "analysis",
        "--calibration", root / "train.csv", "--paired-predictions", root / "wave/predictions.csv")
    summary = json.loads((root / "analysis/summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["num_samples"] == 2
    assert summary["paired_comparisons"][0]["same_model_convention_verified"]
    (root / "SYNTHETIC_FIXTURE_ONLY.txt").write_text(
        "Pipeline integration test with synthetic images. These metrics are NOT results on IIIT5K.\n",
        encoding="utf-8")
    print("Smoke test passed; synthetic fixture metrics must not enter the thesis results.")


if __name__ == "__main__":
    main()
