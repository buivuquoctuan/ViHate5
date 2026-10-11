"""Train a CRNN baseline using a train/validation manifest, never the test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .codec import Alphabet
from .augmentation import AUGMENTATION_SPEC, TrainAugmentation
from .data import BATCH_PADDING_POLICY, WordDataset, collate_words, preprocess_image
from .metrics import edit_distance
from .model import CRNN


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu, cuda[:index], and auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot use a CUDA device")
    return device


def seed_everything(seed: int) -> None:
    # CUDA recurrent kernels require this variable before CUDA initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ctc_minimum_length(text: str) -> int:
    """Repeated neighboring symbols require an intervening blank time step."""
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def validate_ctc_alignment(labels: list[str], input_lengths: Tensor,
                           ids: list[str] | None = None) -> None:
    if len(labels) != len(input_lengths):
        raise ValueError("labels and input_lengths must have the same batch size")
    for i, (label, available) in enumerate(zip(labels, input_lengths.tolist())):
        required = ctc_minimum_length(label)
        if not label or required > available:
            identifier = ids[i] if ids is not None else str(i)
            raise ValueError(
                f"CTC cannot align sample {identifier!r}, label={label!r}: "
                f"requires {required} time steps but has {available}. "
                "Increase --min-width/--max-width or correct the annotation."
            )


AUGMENTATION_CTC_POLICY = "reject_infeasible_original;fallback_original_only_if_augmented_ctc_infeasible"


class ProtectedAugmentedWordDataset(WordDataset):
    """Augment train words while preserving enough horizontal CTC time steps."""

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        original = preprocess_image(image, self.image_height, self.min_width, self.max_width)
        validate_ctc_alignment([row["label"]], torch.tensor([original.shape[-1] // 4]), [row["id"]])
        transformed = self.transform(image) if self.transform is not None else image
        augmented = preprocess_image(transformed, self.image_height, self.min_width, self.max_width)
        fallback = augmented.shape[-1] // 4 < ctc_minimum_length(row["label"])
        tensor = original if fallback else augmented
        return {"image": tensor, "width": tensor.shape[-1], "label": row["label"],
                "id": row["id"], "image_path": row["image_path"],
                "augmentation_fallback": fallback}


def make_datasets(train_path: Path, val_path: Path, preprocessing: dict,
                  augmentation: str) -> tuple[WordDataset, WordDataset]:
    if augmentation == "light":
        train = ProtectedAugmentedWordDataset(train_path, transform=TrainAugmentation(), **preprocessing)
    elif augmentation == "none":
        train = WordDataset(train_path, **preprocessing)
    else:
        raise ValueError(f"Unknown augmentation {augmentation!r}")
    return train, WordDataset(val_path, **preprocessing)


def collate_training_words(samples):
    batch = collate_words(samples)
    batch["augmentation_fallbacks"] = sum(bool(row.get("augmentation_fallback", False)) for row in samples)
    return batch


@dataclass
class SelectionTracker:
    """Track the same validation accuracy/loss rule for saving and early stop."""

    best_accuracy: float = -1.0
    best_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, accuracy: float, loss: float, epoch: int) -> bool:
        if not math.isfinite(accuracy) or not math.isfinite(loss):
            raise FloatingPointError("Validation metrics must be finite")
        improved = (accuracy > self.best_accuracy or
                    (accuracy == self.best_accuracy and loss < self.best_loss))
        if improved:
            self.best_accuracy, self.best_loss, self.best_epoch = accuracy, loss, epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    def should_stop(self, patience: int) -> bool:
        return patience > 0 and self.epochs_without_improvement >= patience


def make_scheduler(optimizer: torch.optim.Optimizer, choice: str):
    if choice == "none":
        return None
    if choice != "plateau":
        raise ValueError(f"Unknown scheduler {choice!r}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)


def initialize_from_checkpoint(model: CRNN, path: Path, alphabet: Alphabet,
                               preprocessing: dict) -> dict[str, Any]:
    """Load compatible model weights; use a fresh optimizer for this run."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Warm-start checkpoint must use format version 2")
    if checkpoint.get("batch_padding") != BATCH_PADDING_POLICY:
        raise ValueError("Warm-start batch padding policy does not match")
    if checkpoint.get("alphabet") != alphabet.characters:
        raise ValueError("Warm-start alphabet does not match")
    if checkpoint.get("preprocessing") != preprocessing:
        raise ValueError("Warm-start preprocessing does not match")
    source_config = checkpoint.get("model_config", {})
    if (source_config.get("num_classes") != model.num_classes or
            source_config.get("hidden_size") != model.hidden_size):
        raise ValueError("Warm-start model architecture does not match")
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("Warm-start state is incompatible with the model architecture") from error
    return {"checkpoint": str(path.resolve()), "checkpoint_sha256": manifest_digest(path),
            "epoch": checkpoint.get("epoch"), "validation_metrics": checkpoint.get("validation_metrics"),
            "model_config": source_config, "preprocessing": checkpoint["preprocessing"],
            "batch_padding": checkpoint["batch_padding"], "source_config": checkpoint.get("config"),
            "optimizer_state_reused": False,
            "dropout_policy": "stateless_dropout_may_change;model_weights_loaded_strictly"}


def encode_targets(labels: list[str], alphabet: Alphabet,
                   device: torch.device) -> tuple[Tensor, Tensor]:
    encoded = [alphabet.encode(label) for label in labels]
    targets = torch.tensor([value for row in encoded for value in row],
                           dtype=torch.long, device=device)
    lengths = torch.tensor([len(row) for row in encoded], dtype=torch.long)
    return targets, lengths


def manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_disjoint_manifests(train_path: Path, val_path: Path) -> None:
    def paths(path: Path) -> set[Path]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {(path.parent / row["image_path"]).resolve()
                    for row in csv.DictReader(handle)}
    overlap = paths(train_path) & paths(val_path)
    if overlap:
        example = str(sorted(overlap, key=str)[0])
        raise ValueError(f"Train/validation image leakage: {len(overlap)} shared paths; {example}")


def run_epoch(model: CRNN, loader: DataLoader, alphabet: Alphabet,
              criterion: nn.CTCLoss, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None,
              grad_clip: float = 5.0) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sample_count = correct = errors = character_count = augmentation_fallbacks = 0
    loss_sum = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            augmentation_fallbacks += batch.get("augmentation_fallbacks", 0)
            images = batch["images"].to(device, non_blocking=True)
            log_probs, input_lengths = model(images, batch["widths"])
            validate_ctc_alignment(batch["labels"], input_lengths, batch["ids"])
            # PyTorch's CUDA CTC backward kernel may be nondeterministic. Run
            # only this loss on CPU; the differentiable copy still propagates
            # gradients to the CNN/BiLSTM on the selected model device.
            targets, target_lengths = encode_targets(batch["labels"], alphabet,
                                                     torch.device("cpu"))
            if not torch.isfinite(log_probs).all():
                raise FloatingPointError(f"Non-finite model outputs for samples {batch['ids']}")
            loss = criterion(log_probs.to("cpu"), targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite CTC loss for samples {batch['ids']}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            predictions = alphabet.decode_greedy(log_probs.detach(), input_lengths)
            count = len(predictions)
            sample_count += count
            loss_sum += loss.item() * count
            for prediction, label in zip(predictions, batch["labels"]):
                correct += prediction == label
                errors += edit_distance(label, prediction)
                character_count += len(label)
    if sample_count == 0:
        raise ValueError("Cannot run an epoch on an empty manifest")
    return {"loss": loss_sum / sample_count, "word_accuracy": correct / sample_count,
            "cer": errors / character_count, "augmentation_fallbacks": augmentation_fallbacks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=["none", "light"], default="none")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many epochs without validation selection improvement; 0 disables")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Warm-start compatible model weights, using a fresh optimizer")
    parser.add_argument("--min-width", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers nonnegative")
    if args.lr <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("lr/grad-clip must be positive and weight-decay nonnegative")
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be nonnegative")
    # Fail before loading data or consuming RNG if this run would overwrite outputs.
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in
           ("best.pt", "history.csv", "config.json", "training_summary.json")):
        raise FileExistsError("Output already contains a run. Choose a new --output directory.")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    alphabet = Alphabet()
    preprocessing = {"image_height": 32, "min_width": args.min_width,
                     "max_width": args.max_width}
    train_path, val_path = args.train.resolve(), args.val.resolve()
    assert_disjoint_manifests(train_path, val_path)
    train_data, val_data = make_datasets(train_path, val_path, preprocessing, args.augmentation)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                      "pin_memory": device.type == "cuda",
                      "worker_init_fn": seed_worker}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator,
                              collate_fn=collate_training_words, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, collate_fn=collate_words, **loader_options)
    model_config = {"num_classes": alphabet.num_classes, "hidden_size": args.hidden_size,
                    "dropout": args.dropout}
    model = CRNN(**model_config).to(device)
    initialization = (initialize_from_checkpoint(model, args.init_checkpoint, alphabet, preprocessing)
                      if args.init_checkpoint is not None else None)
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.scheduler)
    criterion = nn.CTCLoss(blank=alphabet.blank_id, reduction="mean", zero_infinity=False)
    config: dict[str, Any] = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "seed": args.seed, "device": str(device), "torch_version": str(torch.__version__),
        "model_config": model_config, "preprocessing": preprocessing,
        "alphabet": alphabet.characters, "decoder": "greedy_ctc",
        "normalization": "lowercase_alnum_strict",
        "batch_padding": BATCH_PADDING_POLICY,
        "selection_metric": "validation_word_accuracy_then_lower_validation_loss",
        "ctc_loss_device": "cpu", "deterministic_algorithms": True,
        "initialization": initialization,
        "augmentation": {"kind": args.augmentation,
                         "spec": AUGMENTATION_SPEC if args.augmentation == "light" else None,
                         "ctc_policy": AUGMENTATION_CTC_POLICY,
                         "validation": "none"},
        "scheduler": {"kind": args.scheduler, "monitor": "validation_loss",
                      "factor": 0.5, "patience": 5, "min_lr": 1e-5}
                     if scheduler is not None else {"kind": "none"},
        "manifests": {"train": {"path": str(train_path), "sha256": manifest_digest(train_path),
                                 "samples": len(train_data)},
                      "val": {"path": str(val_path), "sha256": manifest_digest(val_path),
                               "samples": len(val_data)}},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fields = ["epoch", "train_loss", "train_word_accuracy", "train_cer", "val_loss",
              "val_word_accuracy", "val_cer", "seconds", "is_best", "lr", "next_lr",
              "best_checkpoint_epoch", "epochs_without_improvement", "stopping_reason",
              "train_augmentation_fallbacks", "val_augmentation_fallbacks"]
    selection = SelectionTracker()
    total_fallbacks = 0
    with (args.output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_metrics = run_epoch(model, train_loader, alphabet, criterion, device,
                                      optimizer, args.grad_clip)
            val_metrics = run_epoch(model, val_loader, alphabet, criterion, device)
            is_best = selection.update(val_metrics["word_accuracy"], val_metrics["loss"], epoch)
            if is_best:
                torch.save({"format_version": 2, "model_state": model.state_dict(),
                            "model_config": model_config, "preprocessing": preprocessing,
                            "batch_padding": BATCH_PADDING_POLICY,
                            "alphabet": alphabet.characters, "epoch": epoch,
                            "best_checkpoint_epoch": epoch,
                            "validation_metrics": val_metrics, "config": config},
                           args.output / "best.pt")
            if scheduler is not None:
                scheduler.step(val_metrics["loss"])
            should_stop = selection.should_stop(args.early_stopping_patience)
            stopping_reason = ("early_stopping" if should_stop else
                               "epochs_completed" if epoch == args.epochs else "running")
            total_fallbacks += train_metrics["augmentation_fallbacks"]
            row = {"epoch": epoch, "seconds": round(time.monotonic() - started, 3),
                   "is_best": int(is_best), "lr": learning_rate,
                   "next_lr": optimizer.param_groups[0]["lr"],
                   "best_checkpoint_epoch": selection.best_epoch,
                   "epochs_without_improvement": selection.epochs_without_improvement,
                   "stopping_reason": stopping_reason,
                   **{f"train_{key}": value for key, value in train_metrics.items()},
                   **{f"val_{key}": value for key, value in val_metrics.items()}}
            writer.writerow(row)
            handle.flush()
            print(f"Epoch {epoch}/{args.epochs}: train loss={train_metrics['loss']:.4f}; "
                  f"val accuracy={val_metrics['word_accuracy']:.2%}; "
                  f"val CER={val_metrics['cer']:.4f}; best={selection.best_accuracy:.2%}; "
                  f"lr={learning_rate:.2g}", flush=True)
            if should_stop:
                break
    summary = {"epochs_completed": epoch, "best_checkpoint_epoch": selection.best_epoch,
               "best_validation_word_accuracy": selection.best_accuracy,
               "best_validation_loss": selection.best_loss, "stopping_reason": stopping_reason,
               "augmentation_fallbacks_total": total_fallbacks,
               "last_epoch_lr": learning_rate, "next_lr": optimizer.param_groups[0]["lr"]}
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {(args.output / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
