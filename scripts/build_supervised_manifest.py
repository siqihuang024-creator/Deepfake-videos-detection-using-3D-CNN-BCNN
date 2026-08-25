"""Derive a supervised-training manifest from the donor-safe one-class manifest.

The one-class manifest parks every fake whose target identity belongs to the
training split in ``unused``, because real-only training never needed them.
Supervised training does, so this script promotes exactly those fakes to
``train`` while keeping the identity contract intact:

- the fake's target identity must be a training identity, and
- the fake's donor identity must also be a training identity.

The second rule mirrors the donor-safe val/test rule. Without it a training
fake built from a val/test donor would leak an evaluation identity into
backpropagation, and the cross-validation numbers would be meaningless.

Real rows and every val/test row are copied unchanged, so a supervised run and
a one-class run are selected and scored on identical videos.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TRAIN_FAKE_SOURCE_SPLITS = ("unused",)


def load_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames)


def train_identities(rows):
    """Identities whose real videos are in the training split, per dataset."""
    identities = {}
    for row in rows:
        if row["split"] == "train" and int(row["label"]) == 1:
            identities.setdefault(row["dataset"], set()).add(row["target_id"])
    if not identities:
        raise ValueError("The source manifest declares no real training videos.")
    return identities


def promote_fake_training_rows(rows, identities):
    """Move donor-safe fakes of training identities from `unused` into `train`."""
    result, counts = [], Counter()
    for row in rows:
        copied = dict(row)
        dataset = copied["dataset"]
        known = identities.get(dataset, set())
        is_candidate = (
            int(copied["label"]) == 0
            and copied["split"] in TRAIN_FAKE_SOURCE_SPLITS
            and copied["target_id"] in known
        )
        if is_candidate:
            donor = copied["donor_id"]
            # An empty donor means the filename carries no second identity, so
            # the clip cannot leak one. Anything else must be a train identity.
            if donor == "" or donor in known:
                copied["split"] = "train"
                counts[(dataset, "promoted")] += 1
            else:
                copied["split"] = "excluded_donor_eval"
                counts[(dataset, "excluded_donor_eval")] += 1
        result.append(copied)
    return result, counts


def summarize(rows, counts):
    videos = Counter((row["dataset"], row["split"], row["class_name"]) for row in rows)
    train_methods = Counter(
        (row["dataset"], row["method"])
        for row in rows
        if row["split"] == "train" and int(row["label"]) == 0
    )
    return {
        "identity_protocol": (
            "Supervised manifest. A fake enters training only when its target AND donor "
            "identities are both training identities; val/test rows are unchanged, so "
            "supervised and one-class runs are evaluated on identical videos."
        ),
        "promotion_counts": {
            "{}:{}".format(dataset, kind): count
            for (dataset, kind), count in sorted(counts.items())
        },
        "videos": {
            "{}:{}:{}".format(dataset, split, name): count
            for (dataset, split, name), count in sorted(videos.items())
        },
        "train_fake_methods": {
            "{}:{}".format(dataset, method): count
            for (dataset, method), count in sorted(train_methods.items())
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="artifacts/manifests/combined_manifest_dfd_donor_safe.csv",
        help="Donor-safe one-class manifest produced by build_manifest.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/manifests/combined_manifest_supervised.csv",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        raise FileNotFoundError("Source manifest not found: {}".format(source))
    rows, fields = load_manifest(source)
    identities = train_identities(rows)
    promoted, counts = promote_fake_training_rows(rows, identities)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(promoted)

    summary = summarize(promoted, counts)
    summary["source_manifest"] = str(source.resolve())
    summary_path = output.with_name(output.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print("Supervised manifest: {}".format(output.resolve()))


if __name__ == "__main__":
    main()
