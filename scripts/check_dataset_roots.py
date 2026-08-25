"""Verify that a manifest resolves against this machine's dataset roots.

Run this first on a rented GPU. A wrong root, a case-mismatched folder, or a
missing video codec otherwise only shows up deep into a training run.
"""

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from video_bcnn.utils import load_config, override_dataset_roots, verify_dataset_roots


def sample_rows(rows, active, per_group):
    """Take a deterministic sample from every dataset/split/class group."""
    groups = collections.defaultdict(list)
    for row in rows:
        if active and row["dataset"] not in active:
            continue
        groups[(row["dataset"], row["split"], row["class_name"])].append(row)
    selected = []
    for key in sorted(groups):
        values = groups[key]
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        selected.extend(rng.sample(values, min(per_group, len(values))))
    return groups, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--per-group", type=int, default=5,
                        help="Videos to check per dataset/split/class group.")
    parser.add_argument("--decode", action="store_true",
                        help="Also open each sampled video to confirm codec support.")
    args = parser.parse_args()

    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    roots = verify_dataset_roots(config)
    print("Dataset roots exist:")
    for name, root in sorted(roots.items()):
        print("  {:<12} {}".format(name, root))

    with open(args.manifest, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    active = set(config["data"].get("active_datasets") or [])
    groups, selected = sample_rows(rows, active, args.per_group)

    print("\nManifest groups (dataset / split / class -> videos):")
    for key in sorted(groups):
        print("  {:<10} {:<22} {:<5} {}".format(key[0], key[1], key[2], len(groups[key])))

    missing, unreadable = [], []
    for row in selected:
        path = Path(roots[row["dataset"]]) / row["path"]
        if not path.is_file():
            missing.append(path)
            continue
        if args.decode:
            import cv2
            capture = cv2.VideoCapture(str(path))
            ok = capture.isOpened() and capture.read()[0]
            capture.release()
            if not ok:
                unreadable.append(path)

    print("\nChecked {} sampled videos.".format(len(selected)))
    if missing:
        print("MISSING {} file(s). First few:".format(len(missing)))
        for path in missing[:5]:
            print("  {}".format(path))
        print("\nThe manifest stores paths relative to the root, so a root that exists but "
              "yields missing files usually means the folder layout or letter case differs "
              "from the Windows copy. Linux is case-sensitive; Windows is not.")
    if unreadable:
        print("UNREADABLE {} file(s) (codec/OpenCV problem). First few:".format(len(unreadable)))
        for path in unreadable[:5]:
            print("  {}".format(path))
    if not missing and not unreadable:
        print("OK: every sampled video resolved{}.".format(" and decoded" if args.decode else ""))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
