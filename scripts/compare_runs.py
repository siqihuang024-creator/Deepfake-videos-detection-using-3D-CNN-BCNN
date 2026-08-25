"""Print the histories of several runs side by side.

Built for the overfit sweep: the question is which architecture change actually
moves the training loss, and that is easiest to see as one table.
"""

import argparse
import csv
import math
from pathlib import Path


def load_history(run_dir):
    path = Path(run_dir) / "logs" / "history.csv"
    if not path.is_file():
        path = Path(run_dir) / "history.csv"
    if not path.is_file():
        raise FileNotFoundError("No history.csv under {}".format(run_dir))
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories to compare.")
    parser.add_argument("--label", nargs="*", default=None,
                        help="Display names, in the same order as the runs.")
    args = parser.parse_args()
    labels = args.label or [Path(run).name for run in args.runs]
    if len(labels) != len(args.runs):
        parser.error("--label must give one name per run.")

    print("{:<32} {:>7} {:>13} {:>13} {:>9} {:>8} {:>12}".format(
        "run", "epochs", "loss first", "loss last", "loss drop", "AUROC", "embed var"))
    print("-" * 100)
    for label, run in zip(labels, args.runs):
        history = load_history(run)
        if not history:
            print("{:<32} {:>7}".format(label[:32], 0))
            continue
        first, last = number(history[0], "train_loss"), number(history[-1], "train_loss")
        drop = float("nan") if not first or math.isnan(first) else 100.0 * (first - last) / abs(first)
        best = max(
            (number(row, "validation_auroc") for row in history),
            default=float("nan"),
        )
        print("{:<32} {:>7} {:>13.4f} {:>13.4f} {:>8.2f}% {:>8.4f} {:>12.4g}".format(
            label[:32], len(history), first, last, drop, best,
            number(history[-1], "embedding_variance_mean"),
        ))
    print("\nloss drop is the headline: a model that cannot reduce its own training "
          "loss on a handful of videos is limited by architecture or optimisation, "
          "not by data.")


if __name__ == "__main__":
    main()
