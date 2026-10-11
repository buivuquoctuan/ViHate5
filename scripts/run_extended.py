"""Run the fixed additional GPU experiments sequentially; never select using test."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 7, 123]
PRETRAIN = {"epochs": 10, "batch_size": 32, "lr": 0.001,
            "dropout": 0.1, "scheduler": "plateau", "seed": 42}
FINETUNE = {"epochs": 50, "batch_size": 16, "lr": 0.0003,
            "dropout": 0.2, "augmentation": "light", "optimizer": "adamw",
            "weight_decay": 0.0001, "scheduler": "plateau",
            "early_stopping_patience": 12}
BASELINE = {"epochs": 50, "batch_size": 16, "lr": 0.001}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def command(module: str, **kwargs) -> list[str]:
    values = [sys.executable, "-m", module]
    for key, value in kwargs.items():
        if value is not None:
            values.extend(["--" + key.replace("_", "-"), str(value)])
    return values


def execute(values: list[str]) -> None:
    print("Running: " + subprocess.list2cmdline(values), flush=True)
    subprocess.run(values, cwd=ROOT, check=True)


def capture_sources(run: Path) -> None:
    files = sorted((ROOT / "src" / "crnn").glob("*.py"))
    files += [Path(__file__), ROOT / "docs" / "ke_hoach_hoan_thien.md"]
    write_json(run / "source_sha256.json", {str(p.relative_to(ROOT)): digest(p) for p in files})
    import torch
    write_json(run / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Skip only fully finished training and evaluation artifacts")
    args = parser.parse_args()
    runs = ROOT / "runs"
    plan_path = runs / "experiment_plan.json"
    plan = {
        "protocol_version": 1, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "fixed_before_additional_test_evaluations",
        "known_baseline_test_accuracy": 671 / 3000,
        "interpretation": "Exploratory follow-on after baseline test was observed",
        "split_metadata_sha256": digest(ROOT / "data/manifests/split_metadata.json"),
        "synthetic_metadata_sha256": digest(ROOT / "data/synthetic_v1/source_metadata.json"),
        "preregistration_document_sha256": digest(ROOT / "docs/ke_hoach_hoan_thien.md"),
        "pretraining": PRETRAIN, "finetuning": FINETUNE, "baseline": BASELINE,
        "seeds": SEEDS, "pretraining_initializations": 1,
        "seed_interpretation": "Three fine-tuning seeds share the same pretraining checkpoint",
        "main_improved_selection": "highest_validation_word_accuracy_then_lower_loss_then_seed",
        "test_use": "Evaluate every seed; do not tune recipes or select seeds using test"}
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError("Experiment plan already exists. Use --resume for finished stages only.")
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        for field in ("pretraining", "finetuning", "baseline", "seeds", "split_metadata_sha256",
                      "synthetic_metadata_sha256", "preregistration_document_sha256"):
            if saved[field] != plan[field]:
                raise ValueError(f"Frozen experiment plan changed: {field}")
    else:
        write_json(plan_path, plan)
    progress_path = runs / "experiment_progress.json"

    def train_stage(name: str, train: str, val: str, options: dict) -> None:
        output = runs / name
        summary = output / "training_summary.json"
        if args.resume and summary.exists() and (output / "best.pt").exists():
            saved = json.loads(summary.read_text(encoding="utf-8"))
            if saved["stopping_reason"] not in {"epochs_completed", "early_stopping"}:
                raise ValueError(f"Incomplete stage cannot be resumed automatically: {name}")
            print(f"Skipping completed stage: {name}", flush=True)
            return
        if any((output / filename).exists() for filename in
               ("best.pt", "history.csv", "config.json", "training_summary.json")):
            raise FileExistsError(
                f"Training stage {name} already contains artifacts. Partial training cannot be resumed; "
                "use a fresh experiment workspace. Existing provenance was kept unchanged.")
        write_json(progress_path, {"phase": "training", "run": name,
                                  "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        output.mkdir(parents=True, exist_ok=True)
        capture_sources(output)
        execute(command("crnn.train", train=train, val=val, output=output,
                        device=args.device, **options))

    train_stage("synthetic_pretrain_v1", "data/synthetic_v1/train.csv",
                "data/synthetic_v1/val.csv", PRETRAIN)
    shared_checkpoint = runs / "synthetic_pretrain_v1/best.pt"
    # Save a domain-transfer check on IIIT validation only, before any new test evaluation.
    transfer = runs / "synthetic_pretrain_v1/iiit_validation"
    if not (args.resume and (transfer / "metrics.json").exists()):
        execute(command("crnn.evaluate", checkpoint=shared_checkpoint,
                        manifest="data/manifests/val.csv", output=transfer,
                        batch_size=16, device=args.device))
    for seed in SEEDS:
        train_stage(f"improved_seed{seed}", "data/manifests/train.csv",
                    "data/manifests/val.csv", {**FINETUNE, "seed": seed,
                                               "init_checkpoint": shared_checkpoint})
    for seed in [7, 123]:
        train_stage(f"baseline_seed{seed}", "data/manifests/train.csv",
                    "data/manifests/val.csv", {**BASELINE, "seed": seed})
    # All training has finished under the frozen plan before these new test runs.
    for name in [f"improved_seed{s}" for s in SEEDS] + ["baseline_seed7", "baseline_seed123"]:
        output = runs / name / "evaluation"
        if args.resume and (output / "metrics.json").exists():
            continue
        write_json(progress_path, {"phase": "evaluation", "run": name,
                                  "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        execute(command("crnn.evaluate", checkpoint=runs / name / "best.pt",
                        manifest="data/manifests/test.csv", output=output,
                        batch_size=16, device=args.device))
    write_json(progress_path, {"phase": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    print("Additional train/evaluation stages complete. Run comparison and robustness next.", flush=True)


if __name__ == "__main__":
    main()
