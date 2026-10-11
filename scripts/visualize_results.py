r"""Render research figures from saved CRNN artifacts; no model inference runs.

Usage: .venv\Scripts\python.exe scripts/visualize_results.py --run runs/baseline_seed42
Missing results are reported and skipped. Every figure records its input hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from crnn.distortions import CORRUPTION_KINDS, CORRUPTION_SPEC, apply_corruption

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KIND_LABELS = {"gaussian_blur": "Gaussian blur", "perspective": "Perspective", "wave": "Wave"}


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite numeric value")
    return result


class Figures:
    def __init__(self, run: Path, manifest: Path | None = None) -> None:
        self.run = run.resolve()
        self.output = self.run / "figures"
        self.manifest = manifest
        self.sources: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.notes: list[str] = []

    def read(self, path: Path) -> bytes:
        content = path.read_bytes()
        self.sources[str(path.resolve())] = {
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
        }
        return content

    def read_csv(self, path: Path) -> list[dict[str, str]]:
        return list(csv.DictReader(io.StringIO(self.read(path).decode("utf-8-sig"))))

    def read_json(self, path: Path) -> dict[str, Any]:
        result = json.loads(self.read(path).decode("utf-8-sig"))
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object: {path}")
        return result

    def save(self, figure: Any, filename: str, details: dict[str, Any]) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.output / filename, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        self.results[filename] = {"status": "generated", **details}
        print(f"Saved: {self.output / filename}")

    def skipped(self, filename: str, reason: str) -> None:
        self.results[filename] = {"status": "skipped", "reason": reason}
        print(f"Skipped {filename}: {reason}")

    def source_image(self, path_text: str, source_csv: Path) -> tuple[Image.Image, Path]:
        path = Path(path_text)
        if not path.is_absolute():
            path = source_csv.parent / path
        content = self.read(path)
        with Image.open(io.BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return image, path.resolve()

    def training(self) -> None:
        path = self.run / "history.csv"
        rows = self.read_csv(path)
        columns = ("train_loss", "val_loss", "train_word_accuracy", "val_word_accuracy", "train_cer", "val_cer")
        valid = []
        for index, row in enumerate(rows, 2):
            try:
                item = {key: number(row[key]) for key in ("epoch", *columns)}
                if item["epoch"] < 1 or not item["epoch"].is_integer():
                    raise ValueError("Invalid epoch")
                if not all(0 <= item[key] <= 1 for key in ("train_word_accuracy", "val_word_accuracy")):
                    raise ValueError("Accuracy outside [0,1]")
                if min(item["train_cer"], item["val_cer"]) < 0:
                    raise ValueError("Negative CER")
                item["is_best"] = row.get("is_best") == "1"
                valid.append(item)
            except (KeyError, TypeError, ValueError):
                self.notes.append(f"Skipped incomplete/invalid history row {index}; a training run may still be writing it.")
        if not valid:
            raise ValueError("No complete recorded training epochs")
        valid.sort(key=lambda row: row["epoch"])
        if len({row["epoch"] for row in valid}) != len(valid):
            raise ValueError("History contains duplicate epochs")
        config_path = self.run / "config.json"
        config = self.read_json(config_path) if config_path.is_file() else {}
        planned = config.get("arguments", {}).get("epochs")
        epochs = [row["epoch"] for row in valid]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        panels = (("loss", "CTC loss (target-length normalized)", 1),
                  ("word_accuracy", "Word accuracy (%)", 100), ("cer", "Micro CER (%)", 100))
        for ax, (key, ylabel, scale) in zip(axes, panels):
            for split, color in (("train", "#245a8d"), ("val", "#db7b2b")):
                ax.plot(epochs, [scale * row[f"{split}_{key}"] for row in valid],
                        label="Train" if split == "train" else "Validation", color=color,
                        linewidth=1.7, marker="o" if len(valid) == 1 else None)
            ax.set(xlabel="Recorded epoch", ylabel=ylabel)
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
            if key == "word_accuracy":
                ax.set_ylim(0, 100)
        fig.suptitle(f"Recorded training history: {len(valid)} {'epoch' if len(valid) == 1 else 'epochs'}" + (f" / {planned} planned" if planned else ""))
        fig.tight_layout()
        self.save(fig, "training_curves.png", {
            "history": str(path.resolve()), "recorded_epochs": [int(value) for value in epochs],
            "planned_epochs": planned, "word_accuracy_and_cer_display_scale": 100,
        })

    def robustness(self) -> None:
        summary_path = self.run / "analysis" / "summary.json"
        pair_path = self.run / "analysis" / "paired_robustness.csv"
        clean = None
        if summary_path.is_file():
            summary = self.read_json(summary_path)
            pairs = summary.get("paired_comparisons", [])
            if summary.get("overall", {}).get("word_accuracy") is not None:
                clean = number(summary["overall"]["word_accuracy"])
        else:
            pairs = []
        if not pairs and pair_path.is_file():
            pairs = self.read_csv(pair_path)
        if not pairs:
            raise ValueError("No measured paired robustness results")
        curves: dict[str, dict[int, float]] = {kind: {} for kind in CORRUPTION_KINDS}
        counts: set[int] = set()
        unverified = []
        for row in pairs:
            kind = row["condition"]
            severity = int(row["severity"])
            baseline = number(row["baseline_word_accuracy"])
            value = number(row["corrupted_word_accuracy"])
            if kind not in curves or severity not in (1, 2, 3):
                raise ValueError("Unexpected corruption condition/severity in paired summary")
            if not 0 <= value <= 1 or not 0 <= baseline <= 1:
                raise ValueError("Robustness accuracy outside [0,1]")
            if clean is None:
                clean = baseline
            if not math.isclose(clean, baseline, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Paired rows do not share the same measured clean baseline")
            if severity in curves[kind]:
                raise ValueError(f"Duplicate robustness measurement: {kind}/{severity}")
            curves[kind][severity] = value
            counts.add(int(row["num_samples"]))
            verified = str(row.get("same_model_convention_verified", "")).lower() == "true"
            if not verified:
                unverified.append(f"{kind}/{severity}")
        if len(counts) != 1:
            raise ValueError("Paired conditions have different sample counts")
        if unverified:
            self.notes.append("Paired provenance not fully verified for: " + ", ".join(unverified))
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for kind, values in curves.items():
            if values:
                levels = [0, *sorted(values)]
                accuracies = [float(clean), *(values[level] for level in sorted(values))]
                ax.plot(levels, np.asarray(accuracies) * 100, marker="o", linewidth=1.8, label=KIND_LABELS[kind])
        ax.set(xlabel="Severity (0 = measured clean baseline)", ylabel="Word accuracy (%)",
               xticks=[0, 1, 2, 3], ylim=(0, 100))
        ax.set_title("Paired robustness on the same images" + (" (provenance incomplete)" if unverified else ""))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        self.save(fig, "robustness_curves.png", {
            "clean_word_accuracy": clean, "num_samples_per_condition": next(iter(counts)),
            "measured_corruptions": curves, "unverified_provenance_conditions": unverified,
            "intervals": "No accuracy or paired-delta intervals drawn in this overview",
        })

    def errors(self) -> None:
        path = self.run / "analysis" / "errors.csv"
        rows = self.read_csv(path)
        rows.sort(key=lambda row: (-int(row["edit_distance"]), -len(row["label"]), row["id"]))
        selected = []
        for row in rows:
            if row["label"] == row["prediction"]:
                continue
            try:
                image, image_path = self.source_image(row["image_path"], path)
            except (OSError, ValueError):
                self.notes.append(f"Missing/unreadable error image: {row.get('id', '<unknown>')}")
                continue
            selected.append((row, image, image_path))
            if len(selected) == 12:
                break
        if not selected:
            raise ValueError("No recorded errors with available images")
        fig, axes = plt.subplots(math.ceil(len(selected) / 3), 3,
                                 figsize=(15, 3.2 * math.ceil(len(selected) / 3)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (row, image, _) in zip(axes.ravel(), selected):
            ax.imshow(image, interpolation="nearest")
            identifier = textwrap.fill(row["id"], width=42)
            prediction = row["prediction"] or "<empty>"
            ax.set_title(f"{identifier}\nGT: {row['label']}  |  Pred: {prediction}\nEdit distance: {row['edit_distance']}", fontsize=9)
        fig.suptitle("Highest-edit-distance errors with available original images")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.save(fig, "error_examples.png", {
            "selection": "Edit distance descending, label length descending, then ID; up to 12 available images",
            "examples": [{"id": row["id"], "image_path": str(image_path), "label": row["label"],
                          "prediction": row["prediction"], "edit_distance": int(row["edit_distance"])}
                         for row, _, image_path in selected],
            "images": "Original crops, EXIF orientation applied for display; source files unchanged",
        })

    def corruptions(self) -> None:
        manifest = self.manifest
        evaluation_path = self.run / "evaluation" / "evaluation_metadata.json"
        if manifest is None and evaluation_path.is_file():
            metadata = self.read_json(evaluation_path)
            if metadata.get("manifest"):
                manifest = Path(metadata["manifest"])
        manifest = manifest or PROJECT_ROOT / "data" / "manifests" / "test.csv"
        rows = self.read_csv(manifest)
        if not rows:
            raise ValueError("Test manifest is empty")
        row = rows[0]
        if row.get("official_split") not in (None, "", "test"):
            raise ValueError("Corruption examples require a test image, not a training image")
        for kind in CORRUPTION_KINDS:
            for severity in (1, 2, 3):
                metadata_path = self.run / "robustness" / f"{kind}_{severity}" / "evaluation_metadata.json"
                if metadata_path.is_file():
                    saved_spec = self.read_json(metadata_path).get("corruption_spec")
                    if saved_spec is not None and saved_spec != CORRUPTION_SPEC:
                        raise ValueError("Current corruption implementation differs from the saved run specification")
        image, image_path = self.source_image(row["image_path"], manifest)
        fig, axes = plt.subplots(3, 4, figsize=(16, 8))
        for index, kind in enumerate(CORRUPTION_KINDS):
            for severity in (0, 1, 2, 3):
                ax = axes[index, severity]
                example = image if severity == 0 else apply_corruption(image, kind, severity)
                ax.imshow(example, interpolation="nearest")
                ax.axis("off")
                ax.set_title("Clean original" if severity == 0 else f"{KIND_LABELS[kind]}: level {severity}", fontsize=10)
        fig.suptitle(f"Controlled transformations of first test image: {row['id']}\nGT: {row['label']} (no predictions or natural-distortion labels generated)")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        self.save(fig, "corruption_examples.png", {
            "test_manifest": str(manifest.resolve()), "id": row["id"], "label": row["label"],
            "image_path": str(image_path), "source_dimensions": list(image.size),
            "corruption_spec": CORRUPTION_SPEC, "transform_stage": "Original image before CRNN preprocessing",
            "note": "Transform illustrations only; no model performance or natural labels inferred",
        })

    def generate(self) -> None:
        jobs = (("training_curves.png", self.training), ("robustness_curves.png", self.robustness),
                ("error_examples.png", self.errors), ("corruption_examples.png", self.corruptions))
        for filename, job in jobs:
            try:
                job()
            except (OSError, ValueError, KeyError, TypeError) as error:
                plt.close("all")
                self.skipped(filename, str(error))
        self.output.mkdir(parents=True, exist_ok=True)
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(self.run),
                    "source_files": self.sources, "figures": self.results, "notes": self.notes,
                    "policy": "Read saved artifacts and source crops only; no training, inference or generated annotations"}
        (self.output / "figure_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/baseline_seed42"))
    parser.add_argument("--manifest", type=Path, help="Optional test manifest for the corruption montage")
    args = parser.parse_args()
    if not args.run.is_dir():
        parser.error(f"Run directory not found: {args.run}")
    Figures(args.run, args.manifest).generate()


if __name__ == "__main__":
    main()
