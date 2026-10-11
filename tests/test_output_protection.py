"""Reject existing artifacts before expensive work or changing provenance."""

import importlib.util
import sys
from pathlib import Path

import pytest

from crnn import evaluate, prepare


def test_evaluation_existing_output_skips_loading_and_keeps_results(tmp_path, monkeypatch):
    saved = tmp_path / "predictions.csv"
    saved.write_text("original results", encoding="utf-8")
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda *args: pytest.fail("Checkpoint should not load"))
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate.main(["--checkpoint", "missing.pt", "--manifest", "missing.csv",
                       "--output", str(tmp_path)])
    assert saved.read_text(encoding="utf-8") == "original results"


def test_preparation_existing_manifest_skips_dataset_scan(tmp_path, monkeypatch):
    saved = tmp_path / "split_metadata.json"
    saved.write_text("original split", encoding="utf-8")
    monkeypatch.setattr(prepare, "load_official_rows", lambda *args: pytest.fail("Must reject before reading images"))
    with pytest.raises(FileExistsError, match="already exist"):
        prepare.main(["--root", "missing", "--output", str(tmp_path)])
    assert saved.read_text(encoding="utf-8") == "original split"


def test_extended_partial_run_keeps_source_snapshot(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/run_extended.py"
    spec = importlib.util.spec_from_file_location("extended_output_test", script)
    extended = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extended)
    monkeypatch.setattr(extended, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script)])
    for relative in ("data/manifests/split_metadata.json", "data/synthetic_v1/source_metadata.json",
                     "docs/ke_hoach_hoan_thien.md"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    run = tmp_path / "runs/synthetic_pretrain_v1"
    run.mkdir(parents=True)
    (run / "config.json").write_text("unfinished run", encoding="utf-8")
    snapshot = run / "source_sha256.json"
    snapshot.write_text("original provenance", encoding="utf-8")
    monkeypatch.setattr(extended, "capture_sources", lambda *args: pytest.fail("Must preserve original provenance"))
    monkeypatch.setattr(extended, "execute", lambda *args: pytest.fail("Must not launch training"))
    with pytest.raises(FileExistsError, match="Partial training"):
        extended.main()
    assert snapshot.read_text(encoding="utf-8") == "original provenance"
    assert not (tmp_path / "runs/experiment_progress.json").exists()
