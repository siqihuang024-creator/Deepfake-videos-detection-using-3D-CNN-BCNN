"""Build identity-aware combined DFD and CelebDFv3 manifests."""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path


DFD_REAL_DIR = "DFD_original sequences"
DFD_FAKE_DIR = "DFD_manipulated_sequences/DFD_manipulated_sequences"
CELEB_REAL_DIR = "REAL"
CELEB_FAKE_DIR = "FAKE"
DFD_ID_PATTERN = re.compile(r"^(\d+)")
DFD_FAKE_PATTERN = re.compile(r"^(\d+)_(\d+)__")
CELEB_REAL_PATTERN = re.compile(r"^(id\d+)_\d+$")
CELEB_FAKE_PATTERN = re.compile(r"^(id\d+)_(id\d+)_\d+$")
CELEB_FAKE_TARGET_PATTERN = re.compile(r"^(id\d+)_")


def split_identities(identity_ids, seed, train_ratio, val_ratio):
    identity_ids = sorted(identity_ids)
    random.Random(seed).shuffle(identity_ids)
    train_end = max(1, int(round(len(identity_ids) * train_ratio)))
    val_end = max(train_end + 1, int(round(len(identity_ids) * (train_ratio + val_ratio))))
    val_end = min(val_end, len(identity_ids) - 1)
    assignments = {}
    for identity_id in identity_ids[:train_end]:
        assignments[identity_id] = "train"
    for identity_id in identity_ids[train_end:val_end]:
        assignments[identity_id] = "val"
    for identity_id in identity_ids[val_end:]:
        assignments[identity_id] = "test"
    return assignments


def make_row(dataset, path, root, label, split, target_id, method,
             donor_id="", donor_seen_in_train=""):
    return {
        "dataset": dataset,
        "path": path.relative_to(root).as_posix(),
        "label": label,
        "class_name": "real" if label else "fake",
        "split": split,
        "source_id": "{}:{}".format(dataset, target_id),
        "target_id": target_id,
        "donor_id": donor_id,
        "donor_seen_in_train": donor_seen_in_train,
        "method": method,
    }


def dfd_rows(root, seed, train_ratio, val_ratio):
    real_paths = sorted((root / DFD_REAL_DIR).glob("*.mp4"))
    fake_paths = sorted((root / DFD_FAKE_DIR).glob("*.mp4"))
    if not real_paths or not fake_paths:
        raise FileNotFoundError("DFD MP4 files were not found below {}".format(root))
    real_ids = {DFD_ID_PATTERN.match(path.name).group(1) for path in real_paths}
    assignments = split_identities(real_ids, seed, train_ratio, val_ratio)
    rows = []
    for path in real_paths:
        target_id = DFD_ID_PATTERN.match(path.name).group(1)
        rows.append(make_row("DFD", path, root, 1, assignments[target_id], target_id, "real"))
    train_ids = {identity for identity, split in assignments.items() if split == "train"}
    for path in fake_paths:
        match = DFD_FAKE_PATTERN.match(path.name)
        if match is None:
            raise ValueError("Unexpected DFD fake filename: {}".format(path.name))
        target_id, donor_id = match.groups()
        split = assignments.get(target_id, "unused")
        if split == "train":
            split = "unused"
        rows.append(make_row(
            "DFD", path, root, 0, split, target_id, "DeepFakeDetection",
            donor_id, str(donor_id in train_ids).lower(),
        ))
    return rows, assignments


def donor_safe_dfd(rows, assignments):
    """Exclude DFD val/test fakes whose donor appeared in DFD real training."""
    train_ids = {identity for identity, split in assignments.items() if split == "train"}
    result = []
    for item in rows:
        copied = dict(item)
        if (
            copied["dataset"] == "DFD"
            and int(copied["label"]) == 0
            and copied["split"] in ("val", "test")
            and copied["donor_id"] in train_ids
        ):
            copied["split"] = "excluded_donor_train"
        result.append(copied)
    return result


def celeb_rows(root, seed, train_ratio, val_ratio):
    real_root, fake_root = root / CELEB_REAL_DIR, root / CELEB_FAKE_DIR
    real_paths = sorted(real_root.rglob("*.mp4"))
    fake_paths = sorted(fake_root.rglob("*.mp4"))
    if not real_paths or not fake_paths:
        raise FileNotFoundError("CelebDFv3 MP4 files were not found below {}".format(root))

    def real_source_id(path):
        match = CELEB_REAL_PATTERN.match(path.stem)
        if match:
            return match.group(1)
        if path.parent.name == "Real_YouTube" and path.stem.isdigit():
            return "youtube_{}".format(path.stem)
        raise ValueError("Unexpected CelebDFv3 real filename: {}".format(path.name))

    real_ids = {real_source_id(path) for path in real_paths}
    assignments = split_identities(real_ids, seed, train_ratio, val_ratio)
    rows = []
    for path in real_paths:
        target_id = real_source_id(path)
        rows.append(make_row(
            "CelebDFv3", path, root, 1, assignments[target_id], target_id,
            "real/{}".format(path.parent.name),
        ))
    for path in fake_paths:
        match = CELEB_FAKE_PATTERN.match(path.stem)
        if match:
            target_id, donor_id = match.groups()
        else:
            target_match = CELEB_FAKE_TARGET_PATTERN.match(path.stem)
            if target_match is None:
                raise ValueError("Unexpected CelebDFv3 fake filename: {}".format(path.name))
            target_id, donor_id = target_match.group(1), ""
        if target_id not in assignments:
            raise ValueError("CelebDFv3 fake target has no real identity: {}".format(path.name))
        split = assignments[target_id]
        if split == "train":
            split = "unused"
        method = "/".join(path.relative_to(fake_root).parts[:-1]) or "CelebDFv3"
        rows.append(make_row(
            "CelebDFv3", path, root, 0, split, target_id, method, donor_id,
        ))
    return rows, assignments


def summarize(rows, assignments):
    video_counts = Counter((item["dataset"], item["split"], item["class_name"]) for item in rows)
    method_counts = Counter((item["dataset"], item["method"]) for item in rows if item["label"] == 0)
    return {
        "identity_protocol": (
            "Dataset-local identity splits. DFD val/test fakes are donor-safe in the strict manifest; "
            "fake videos are never assigned to training."
        ),
        "identities": {
            dataset: {
                split: sum(value == split for value in values.values())
                for split in ("train", "val", "test")
            }
            for dataset, values in assignments.items()
        },
        "videos": {
            "{}:{}:{}".format(dataset, split, label): count
            for (dataset, split, label), count in sorted(video_counts.items())
        },
        "fake_methods": {
            "{}:{}".format(dataset, method): count
            for (dataset, method), count in sorted(method_counts.items())
        },
    }


def write_manifest(path, rows):
    fields = [
        "dataset", "path", "label", "class_name", "split", "source_id",
        "target_id", "donor_id", "donor_seen_in_train", "method",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfd-root", required=True)
    parser.add_argument("--celeb-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    dfd_root, celeb_root = Path(args.dfd_root).resolve(), Path(args.celeb_root).resolve()
    dfd, dfd_assignments = dfd_rows(dfd_root, args.seed, args.train_ratio, args.val_ratio)
    celeb, celeb_assignments = celeb_rows(celeb_root, args.seed, args.train_ratio, args.val_ratio)
    rows = dfd + celeb
    strict_rows = donor_safe_dfd(rows, dfd_assignments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "combined_manifest.csv", rows)
    write_manifest(output_dir / "combined_manifest_dfd_donor_safe.csv", strict_rows)
    assignments = {"DFD": dfd_assignments, "CelebDFv3": celeb_assignments}
    for name, values in (("combined_summary.json", rows), ("combined_summary_dfd_donor_safe.json", strict_rows)):
        summary = summarize(values, assignments)
        summary["roots"] = {"DFD": str(dfd_root), "CelebDFv3": str(celeb_root)}
        with open(output_dir / name, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    print(json.dumps(summarize(strict_rows, assignments), indent=2))


if __name__ == "__main__":
    main()
