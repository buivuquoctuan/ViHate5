"""Convert the official IIIT5K MATLAB annotations and freeze train/val/test manifests."""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

from .codec import Alphabet, minimum_ctc_steps
from .data import preprocess_image


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matlab_string(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in "US":
        return "".join(str(item) for item in value.ravel())
    raise ValueError(f"Not a MATLAB string: {type(value)}")


def record_fields(record):
    if hasattr(record, "_fieldnames"):
        return {name.lower(): getattr(record, name) for name in record._fieldnames}
    if isinstance(record, np.void) and record.dtype.names:
        return {name.lower(): record[name] for name in record.dtype.names}
    raise ValueError("Expected MATLAB struct records with ImgName and GT fields.")


def load_official_rows(root, split):
    matches = [p for p in root.iterdir() if p.name.lower() == f"{split}data.mat"]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected {split}data.mat in {root}")
    annotation = matches[0]
    data = loadmat(annotation, squeeze_me=True, struct_as_record=False)
    keys = [key for key in data if key.lower() == f"{split}data"]
    if len(keys) != 1:
        raise ValueError(f"Expected variable {split}Data in {annotation}")
    alphabet = Alphabet()
    rows = []
    for record in np.asarray(data[keys[0]], dtype=object).ravel():
        fields = record_fields(record)
        image_name = matlab_string(fields["imgname"]).replace("\\", "/")
        ground_truth_key = next((name for name in ("groundtruth", "gt") if name in fields), None)
        if ground_truth_key is None:
            raise ValueError(f"Missing GroundTruth/GT field in {annotation}")
        original = matlab_string(fields[ground_truth_key])
        label = original.lower()
        alphabet.encode(label)  # Fail rather than silently stripping/skipping labels.
        image_path = (root / image_name).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ValueError(f"Invalid or missing image: {image_name}")
        with Image.open(image_path) as image:
            image.verify()
        rows.append({
            "id": image_name, "image_path": str(image_path), "label": label,
            "original_label": original, "official_split": split,
        })
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in official {split} annotations.")
    return rows, annotation


def grouped_split(rows, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one.")
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    labels = sorted(groups)
    random.Random(seed).shuffle(labels)
    target = max(1, round(len(rows) * val_fraction))
    # Subset-sum finds the target size without dividing a normalized word group.
    reachable = {0: None}
    for label in labels:
        size = len(groups[label])
        for total in sorted(list(reachable), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in reachable:
                reachable[new_total] = (total, label)
        if target in reachable:
            break
    total = max(reachable)
    if total == 0 or total == len(rows):
        raise ValueError("Cannot create nonempty grouped validation split.")
    selected = set()
    while total:
        total, label = reachable[total]
        selected.add(label)
    train = [row for row in rows if row["label"] not in selected]
    val = [row for row in rows if row["label"] in selected]
    return train, val


def write_manifest(path, rows):
    columns = ["id", "image_path", "label", "original_label", "official_split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            copy = dict(row)
            copy["image_path"] = Path(os.path.relpath(row["image_path"], path.parent)).as_posix()
            writer.writerow(copy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/IIIT5K"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-width", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--allow-nonstandard-size", action="store_true", help="For test fixtures only.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    if not args.overwrite and any((output / name).exists() for name in
                                  ("train.csv", "val.csv", "test.csv", "split_metadata.json")):
        raise FileExistsError("Manifests already exist; use --overwrite only to intentionally regenerate.")
    full_train, train_mat = load_official_rows(root, "train")
    test, test_mat = load_official_rows(root, "test")
    if not args.allow_nonstandard_size and (len(full_train), len(test)) != (2000, 3000):
        raise ValueError(f"Official counts must be 2000/3000, found {len(full_train)}/{len(test)}.")
    if {row["id"] for row in full_train} & {row["id"] for row in test}:
        raise ValueError("Train and test image IDs overlap.")
    train, val = grouped_split(full_train, args.val_fraction, args.seed)
    hashes = {name: {file_hash(row["image_path"]) for row in rows}
              for name, rows in [("train", train), ("val", val), ("test", test)]}
    if hashes["train"] & hashes["val"]:
        raise ValueError("Identical image bytes cross train/val; revise grouping before training.")
    feasibility = {}
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        height_reduced = 0
        for row in rows:
            with Image.open(row["image_path"]) as image:
                tensor = preprocess_image(image, min_width=args.min_width, max_width=args.max_width)
                height_reduced += int(image.width * 32 / image.height > args.max_width)
            steps = tensor.shape[-1] // 4
            required = minimum_ctc_steps(row["label"])
            if steps < required:
                raise ValueError(f"CTC infeasible {row['id']}: T={steps} < {required}; increase min/max width.")
        feasibility[name] = {"all_ctc_feasible": True, "height_reduced_to_fit_width": height_reduced}
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{name}.csv" for name in ["train", "val", "test"]]
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Manifests already exist; use --overwrite only to intentionally regenerate.")
    for path, rows in zip(paths, [train, val, test]):
        write_manifest(path, rows)
    annotation_path = output / "test_annotations.csv"
    if not annotation_path.exists():
        with annotation_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "blur_manual", "distortion", "notes"])
            writer.writeheader()
            writer.writerows({"id": row["id"], "blur_manual": "", "distortion": "", "notes": ""} for row in test)
    metadata = {
        "seed": args.seed, "val_fraction": args.val_fraction,
        "split_method": "normalized-word-grouped subset-sum validation",
        "counts": {"official_train": len(full_train), "train": len(train), "val": len(val), "test": len(test)},
        "label_policy": "lowercase only; strictly 0-9,a-z; no sample filtering",
        "train_val_word_overlap": len({r["label"] for r in train} & {r["label"] for r in val}),
        "image_hash_overlap": {"train_val": len(hashes["train"] & hashes["val"]),
                               "train_test": len(hashes["train"] & hashes["test"]),
                               "val_test": len(hashes["val"] & hashes["test"])},
        "annotation_sha256": {train_mat.name: file_hash(train_mat), test_mat.name: file_hash(test_mat)},
        "preprocessing": {"image_height": 32, "min_width": args.min_width, "max_width": args.max_width},
        "audit": feasibility,
        "manifest_sha256": {path.name: file_hash(path) for path in paths},
    }
    (output / "split_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
