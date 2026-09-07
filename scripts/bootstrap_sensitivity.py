"""How much of a reported AUROC interval is the resampling scheme's choice?

Resampling the same scores by video and by identity gives different intervals,
and the difference is easy to over-read. Two cautions are built into what this
prints.

First, the two arms differ by more than the unit. Both go through
``clustered_auroc_interval``, which resamples groups without stratifying by
class and takes a percentile interval; passing singleton groups makes the first
arm a non-stratified row bootstrap, which is not the stratified case/control
bootstrap conventional in ROC analysis. Under a 169:5431 split, declining to
stratify widens an interval on its own.

Second, the cluster count that matters is not the number of identities but the
number carrying each class. CelebDF++'s test split has 57 target identities and
5431 forgeries, but the forgeries come from twelve of them; DFD's has four
identities of which three carry forgeries. A percentile interval over three or
twelve effective clusters is not a precise interval, and a narrow one is not
evidence of precision.

So this reports coverage of chance, never significance.

Usage:
    python scripts/bootstrap_sensitivity.py \
        --scores results/run_stage_a_dfd_decimate/reports/test_scores.csv \
        --scores results/run_stage_a_celebdfv3_face_stageb_celeb/report_test_scores.csv \
        --output results/diagnostics/bootstrap_sensitivity.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_bcnn.metrics import clustered_auroc_interval  # noqa: E402


def summarise(path, draws, seed):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label_real"]) for row in rows])
    scores = np.asarray([float(row["anomaly_score"]) for row in rows])
    identities = [row["target_id"] for row in rows]
    fake = 1 - labels

    real_ids = {identity for identity, is_real in zip(identities, labels) if is_real}
    fake_ids = {identity for identity, is_real in zip(identities, labels) if not is_real}

    per_video = clustered_auroc_interval(
        fake, scores, [str(i) for i in range(len(rows))], draws=draws, seed=seed)
    per_identity = clustered_auroc_interval(
        fake, scores, identities, draws=draws, seed=seed)
    for entry in (per_video, per_identity):
        entry["width"] = entry["high"] - entry["low"]
        entry["covers_chance"] = bool(entry["low"] <= 0.5 <= entry["high"])

    return {
        "scores": str(Path(path).as_posix()),
        "videos": len(rows),
        "real": int(labels.sum()),
        "fake": int(fake.sum()),
        "identities_total": len(set(identities)),
        "identities_with_real": len(real_ids),
        "identities_with_fake": len(fake_ids),
        "effective_clusters": min(len(real_ids), len(fake_ids)),
        "per_video_bootstrap": per_video,
        "per_identity_bootstrap": per_identity,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", action="append", required=True,
                        help="A *_scores.csv; repeatable.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = [summarise(path, args.draws, args.seed) for path in args.scores]
    for record in records:
        print("== {}".format(record["scores"]))
        print("   {} videos, {} real / {} fake".format(
            record["videos"], record["real"], record["fake"]))
        print("   identities: {} total, {} carry real, {} carry fake -> {} effective "
              "clusters".format(record["identities_total"], record["identities_with_real"],
                                record["identities_with_fake"], record["effective_clusters"]))
        for label, key in (("per video   ", "per_video_bootstrap"),
                           ("per identity", "per_identity_bootstrap")):
            entry = record[key]
            print("   {}  AUROC {:.4f}  [{:.3f}, {:.3f}]  width {:.3f}  "
                  "covers 0.50: {}  draws kept {}".format(
                      label, entry["auroc"], entry["low"], entry["high"],
                      entry["width"], entry["covers_chance"], entry["draws"]))
        print()

    destination = Path(args.output or (ROOT / "results/diagnostics"
                                       / "bootstrap_sensitivity.json"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump({"draws": args.draws, "seed": args.seed, "runs": records},
                  handle, indent=2)
    print("wrote {}".format(destination))
    print("Neither arm is the stratified case/control bootstrap conventional in ROC "
          "analysis, and the effective cluster counts above are small. Read these as "
          "a sensitivity analysis, not as a significance test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
