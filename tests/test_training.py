"""Behavioral regression checks for safe augmentation and training controls."""

import csv
import hashlib
import json
import random

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from torch import nn

import crnn.augmentation as augmentation_module
import crnn.train as training
from crnn.augmentation import TrainAugmentation
from crnn.codec import Alphabet
from crnn.data import BATCH_PADDING_POLICY, WordDataset, preprocess_image
from crnn.model import CRNN
from crnn.train import (
    ProtectedAugmentedWordDataset,
    SelectionTracker,
    build_parser,
    collate_training_words,
    initialize_from_checkpoint,
    make_datasets,
    make_scheduler,
)


def word_image(width=64, height=32):
    """A varied image makes changes in contrast/noise visible without a font."""
    columns = np.linspace(30, 220, width, dtype=np.uint8)
    pixels = np.tile(columns, (height, 1))
    pixels[height // 4:3 * height // 4, width // 4:width // 2] = 15
    return Image.fromarray(pixels).convert("RGB")


def write_manifest(directory, name, image=None, label="book"):
    image = image if image is not None else word_image()
    image_path = directory / f"{name}.png"
    image.save(image_path)
    path = directory / f"{name}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "label"])
        writer.writeheader()
        writer.writerow({"id": name, "image_path": image_path.name, "label": label})
    return path


def test_augmentation_changes_training_but_never_validation(tmp_path, monkeypatch):
    source = word_image()
    train_path = write_manifest(tmp_path, "train", source)
    val_path = write_manifest(tmp_path, "val", source)
    preprocessing = {"image_height": 32, "min_width": 4, "max_width": 64}
    calls = []

    class InvertImage:
        def __call__(self, image):
            calls.append(image.size)
            return ImageOps.invert(image)

    monkeypatch.setattr(training, "TrainAugmentation", InvertImage)
    train_data, val_data = make_datasets(train_path, val_path, preprocessing, "light")
    train_sample, val_sample = train_data[0], val_data[0]
    assert calls == [source.size]
    assert train_sample["augmentation_fallback"] is False
    assert torch.equal(train_sample["image"], preprocess_image(ImageOps.invert(source), **preprocessing))
    assert torch.equal(val_sample["image"], preprocess_image(source, **preprocessing))
    assert not torch.equal(train_sample["image"], val_sample["image"])
    # Every subsequent validation access stays independent of train randomness.
    assert torch.equal(val_sample["image"], val_data[0]["image"])
    assert calls == [source.size]
    plain_train, plain_val = make_datasets(train_path, val_path, preprocessing, "none")
    assert torch.equal(plain_train[0]["image"], plain_val[0]["image"])


def test_repeated_label_falls_back_if_augmentation_loses_ctc_steps(tmp_path):
    image = word_image(width=20)
    manifest = write_manifest(tmp_path, "repeated", image, label="aaa")
    preprocessing = {"image_height": 32, "min_width": 4, "max_width": 64}
    dataset = ProtectedAugmentedWordDataset(
        manifest, transform=lambda source: source.resize((20, 320)), **preprocessing)
    sample = dataset[0]
    # aaa needs five steps; the original has five and the tall transform has one.
    assert sample["width"] == 20
    assert sample["augmentation_fallback"] is True
    assert torch.equal(sample["image"], preprocess_image(image, **preprocessing))
    original_sample = WordDataset(manifest, **preprocessing)[0]
    batch = collate_training_words([sample, original_sample])
    assert batch["augmentation_fallbacks"] == 1
    assert batch["widths"].tolist() == [20, 20]
    assert batch["labels"] == ["aaa", "aaa"]


def test_infeasible_original_is_rejected_before_transform_can_rescue_it(tmp_path):
    manifest = write_manifest(tmp_path, "too_short", word_image(width=16), label="aaa")
    calls = []

    def widen(image):
        calls.append(image.size)
        return image.resize((64, 32))

    dataset = ProtectedAugmentedWordDataset(
        manifest, min_width=4, max_width=64, transform=widen)
    with pytest.raises(ValueError, match="requires 5 time steps but has 4"):
        dataset[0]
    assert calls == []


def test_training_augmentation_repeats_with_both_rng_seeds():
    image = word_image()

    def augmented(python_seed, numpy_seed):
        random.seed(python_seed)
        np.random.seed(numpy_seed)
        return np.asarray(TrainAugmentation()(image))

    # Seed 7 exercises blur and NumPy noise with the actual event probabilities.
    first = augmented(7, 19)
    assert np.array_equal(first, augmented(7, 19))
    assert not np.array_equal(first, augmented(7, 20))
    assert not np.array_equal(first, augmented(8, 19))


def test_rotation_expands_canvas_instead_of_cropping(monkeypatch):
    draws = iter([1.0, 1.0, 0.0])  # Skip blur/noise, apply rotation.
    monkeypatch.setattr(augmentation_module.random, "random", lambda: next(draws))
    monkeypatch.setattr(
        augmentation_module.random, "uniform",
        lambda low, high: 3.0 if low < 0 else 1.0)
    image = word_image(width=80)
    rotated = TrainAugmentation()(image)
    assert rotated.width >= image.width
    assert rotated.height > image.height
    assert rotated.getpixel((0, 0)) == (255, 255, 255)


def test_selection_accuracy_then_loss_controls_early_stop():
    tracker = SelectionTracker()
    assert tracker.update(0.4, 2.0, 1)
    # A much lower loss cannot compensate for lower word accuracy.
    assert not tracker.update(0.39, 0.1, 2)
    assert not tracker.update(0.4, 2.0, 3)
    assert tracker.best_epoch == 1
    assert tracker.epochs_without_improvement == 2
    assert tracker.should_stop(2)
    assert not tracker.should_stop(0)
    assert tracker.update(0.4, 1.9, 4)
    assert tracker.epochs_without_improvement == 0
    assert not tracker.should_stop(2)
    # Higher accuracy wins even when validation loss increases.
    assert tracker.update(0.5, 3.0, 5)
    assert (tracker.best_accuracy, tracker.best_loss, tracker.best_epoch) == (0.5, 3.0, 5)


@pytest.mark.parametrize("accuracy,loss", [(float("nan"), 1.0), (0.5, float("inf"))])
def test_selection_rejects_nonfinite_validation(accuracy, loss):
    with pytest.raises(FloatingPointError, match="finite"):
        SelectionTracker().update(accuracy, loss, 1)


def test_plateau_scheduler_waits_for_loss_plateau_and_respects_minimum_lr():
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=8e-5)
    assert make_scheduler(optimizer, "none") is None
    scheduler = make_scheduler(optimizer, "plateau")
    scheduler.step(2.0)
    for _ in range(5):
        scheduler.step(3.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(8e-5)
    scheduler.step(1.0)  # Lower validation loss resets patience.
    for _ in range(5):
        scheduler.step(2.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(8e-5)
    scheduler.step(2.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4e-5)
    for expected_lr in (2e-5, 1e-5, 1e-5):
        for _ in range(6):
            scheduler.step(2.0)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


@pytest.fixture(scope="module")
def warm_start_checkpoint(tmp_path_factory):
    """Save the large CNN once; metadata rejection tests reuse its tensors."""
    alphabet = Alphabet()
    preprocessing = {"image_height": 32, "min_width": 64, "max_width": 320}
    torch.manual_seed(31)
    source = CRNN(alphabet.num_classes, hidden_size=8)
    with torch.no_grad():
        source.classifier.bias.copy_(torch.arange(alphabet.num_classes, dtype=torch.float32))
    checkpoint = {
        "format_version": 2,
        "batch_padding": BATCH_PADDING_POLICY,
        "alphabet": alphabet.characters,
        "preprocessing": preprocessing,
        # Original format-2 runs predate the optional stateless dropout field.
        "model_config": {"num_classes": alphabet.num_classes, "hidden_size": 8},
        "model_state": source.state_dict(),
        "epoch": 12,
        "validation_metrics": {"word_accuracy": 0.7, "loss": 1.0},
        "config": {"seed": 31},
        "optimizer_state": {"source_only": True},
    }
    path = tmp_path_factory.mktemp("warm_start") / "source.pt"
    torch.save(checkpoint, path)
    return path, checkpoint, alphabet, preprocessing


def test_warm_start_loads_weights_with_new_dropout_and_fresh_optimizer(warm_start_checkpoint):
    path, checkpoint, alphabet, preprocessing = warm_start_checkpoint
    target = CRNN(alphabet.num_classes, hidden_size=8, dropout=0.3)
    optimizer = torch.optim.Adam(target.parameters(), lr=1e-4)
    provenance = initialize_from_checkpoint(target, path, alphabet, preprocessing)
    assert target.dropout == pytest.approx(0.3)
    assert optimizer.state_dict()["state"] == {}
    assert provenance["optimizer_state_reused"] is False
    assert provenance["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["epoch"] == 12
    assert provenance["validation_metrics"] == checkpoint["validation_metrics"]
    assert provenance["source_config"] == {"seed": 31}
    actual = target.state_dict()
    assert actual.keys() == checkpoint["model_state"].keys()
    for key, expected in checkpoint["model_state"].items():
        assert torch.equal(actual[key], expected), key


@pytest.mark.parametrize("mismatch,expected_message", [
    ("format", "format version"),
    ("padding", "padding"),
    ("alphabet", "alphabet"),
    ("preprocessing", "preprocessing"),
    ("num_classes", "architecture"),
    ("hidden_size", "architecture"),
    ("missing_state", "state"),
    ("unexpected_state", "state"),
])
def test_warm_start_rejects_incompatible_checkpoint(
        warm_start_checkpoint, monkeypatch, mismatch, expected_message):
    path, original, alphabet, preprocessing = warm_start_checkpoint
    checkpoint = dict(original)
    if mismatch == "format":
        checkpoint["format_version"] = 1
    elif mismatch == "padding":
        checkpoint["batch_padding"] = "legacy"
    elif mismatch == "alphabet":
        checkpoint["alphabet"] = alphabet.characters[::-1]
    elif mismatch == "preprocessing":
        checkpoint["preprocessing"] = {**preprocessing, "max_width": 640}
    elif mismatch in {"num_classes", "hidden_size"}:
        checkpoint["model_config"] = {
            **original["model_config"], mismatch: original["model_config"][mismatch] + 1}
    else:
        checkpoint["model_state"] = dict(original["model_state"])
        if mismatch == "missing_state":
            checkpoint["model_state"].pop("classifier.bias")
        else:
            checkpoint["model_state"]["unexpected.parameter"] = torch.zeros(1)
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: checkpoint)
    target = CRNN(alphabet.num_classes, hidden_size=8)
    with pytest.raises(ValueError, match=expected_message):
        initialize_from_checkpoint(target, path, alphabet, preprocessing)


def test_new_cli_defaults_preserve_baseline_behavior():
    args = build_parser().parse_args(["--train", "train.csv", "--val", "val.csv"])
    assert args.dropout == 0.0
    assert args.augmentation == "none"
    assert args.optimizer == "adam"
    assert args.scheduler == "none"
    assert args.early_stopping_patience == 0
    assert args.init_checkpoint is None


def test_cli_uses_validation_loss_and_saves_best_epoch_when_stopping(tmp_path, monkeypatch):
    """Exercise orchestration using controlled metrics, without a GPU trial."""
    train_path = write_manifest(tmp_path, "train")
    val_path = write_manifest(tmp_path, "val")
    output = tmp_path / "run"
    scheduler_values = []
    validation = iter([
        (0.5, 2.0), (0.5, 1.0), (0.4, 0.5), (0.5, 1.5),
    ])

    class TinyModel(nn.Module):
        def __init__(self, **config):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    class RecordingScheduler:
        def step(self, metric):
            scheduler_values.append(metric)

    def fake_epoch(model, loader, alphabet, criterion, device, optimizer=None, grad_clip=5.0):
        assert device.type == "cpu"
        if optimizer is not None:
            assert isinstance(optimizer, torch.optim.AdamW)
            assert isinstance(loader.dataset, ProtectedAugmentedWordDataset)
            return {"loss": 9.0, "word_accuracy": 0.0, "cer": 1.0, "augmentation_fallbacks": 1}
        assert type(loader.dataset) is WordDataset
        accuracy, loss = next(validation)
        return {"loss": loss, "word_accuracy": accuracy, "cer": 0.3, "augmentation_fallbacks": 0}

    monkeypatch.setattr(training, "CRNN", TinyModel)
    monkeypatch.setattr(training, "run_epoch", fake_epoch)
    monkeypatch.setattr(training, "make_scheduler", lambda *args: RecordingScheduler())
    training.main([
        "--train", str(train_path), "--val", str(val_path), "--output", str(output),
        "--device", "cpu", "--epochs", "10", "--early-stopping-patience", "2",
        "--augmentation", "light", "--optimizer", "adamw", "--scheduler", "plateau",
    ])
    assert scheduler_values == [2.0, 1.0, 0.5, 1.5]
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == checkpoint["best_checkpoint_epoch"] == 2
    assert checkpoint["validation_metrics"]["loss"] == 1.0
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_completed"] == 4
    assert summary["best_checkpoint_epoch"] == 2
    assert summary["stopping_reason"] == "early_stopping"
    assert summary["augmentation_fallbacks_total"] == 4
    with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert [row["is_best"] for row in history] == ["1", "1", "0", "0"]
    assert history[-1]["epochs_without_improvement"] == "2"
    assert history[-1]["stopping_reason"] == "early_stopping"


def test_real_cpu_training_updates_weights_and_serializes_run(tmp_path):
    train_path = write_manifest(tmp_path, "train", label="book")
    second_image = tmp_path / "train_second.png"
    word_image(width=48).save(second_image)
    with train_path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["train_second", second_image.name, "cat"])
    val_path = write_manifest(tmp_path, "val", label="ab")
    output = tmp_path / "real_run"
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(42)
        initial_model = CRNN(Alphabet().num_classes, hidden_size=8, dropout=0.2)
        initial_weight = initial_model.cnn[0].weight.detach().clone()
        del initial_model
        training.main([
            "--train", str(train_path), "--val", str(val_path), "--output", str(output),
            "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--hidden-size", "8",
            "--min-width", "16", "--max-width", "64", "--dropout", "0.2",
            "--augmentation", "light", "--optimizer", "adamw", "--scheduler", "plateau",
        ])
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    finally:
        torch.set_num_threads(original_threads)
    assert not torch.equal(checkpoint["model_state"]["cnn.0.weight"], initial_weight)
    assert checkpoint["model_config"]["dropout"] == pytest.approx(0.2)
    assert checkpoint["epoch"] == 1
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["device"] == "cpu"
    assert config["manifests"]["train"]["samples"] == 2
    assert config["manifests"]["val"]["samples"] == 1
    assert config["augmentation"]["validation"] == "none"
    assert config["initialization"] is None
    with (output / "history.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    assert len(history) == 1
    row = history[0]
    assert np.isfinite(float(row["train_loss"]))
    assert np.isfinite(float(row["val_loss"]))
    assert float(row["lr"]) == float(row["next_lr"]) == pytest.approx(1e-3)
    assert row["best_checkpoint_epoch"] == "1"
    assert row["stopping_reason"] == "epochs_completed"
    summary = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["epochs_completed"] == summary["best_checkpoint_epoch"] == 1
    assert summary["stopping_reason"] == "epochs_completed"
