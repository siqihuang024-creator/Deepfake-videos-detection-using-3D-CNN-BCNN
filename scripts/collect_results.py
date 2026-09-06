"""Harvest every run's numbers out of artifacts/ into a tracked results/ tree.

`artifacts/*` is gitignored, so per-epoch curves, evaluation reports and
checkpoints live only on the rented workstation. Returning the pod deletes the
figures a paper needs, and hand-copying selected epochs into a prose log -- what
has been happening -- keeps the epochs someone happened to paste and loses the
rest.

What this copies is small and complete: `logs/history.json` and
`logs/history.csv` per run (tens of kilobytes for sixty epochs), plus any
evaluation reports. Checkpoints stay behind; they are far too large for git and
are the one thing that has to be backed up separately if a run is ever to be
re-evaluated without retraining.

Two derived files are the point of the exercise:

  results/curves.csv   long format, one row per (run, epoch) -- what a plotting
                       script consumes directly, no reshaping
  results/summary.csv  one row per run, with the settings that distinguish it
                       beside the numbers it produced -- the results table

Settings come from the checkpoint, which is where both trainers record the
config that actually ran after command-line overrides. That matters here: runs
have been launched with flags the config file never carried.

Usage:
    python scripts/collect_results.py            # then git add results/
    python scripts/collect_results.py --runs run_stage_a_dfd_decimate_full60
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Config keys worth carrying into the summary: the ones that differ between
# runs and therefore explain their numbers.
DATA_KEYS = ("frame_mode", "face_crop", "face_margin", "crop_padding",
             "decimate_step", "clip_length", "train_samples_per_dataset_per_epoch",
             "align_landmarks")
MODEL_KEYS = ("activation", "norm", "spatial_pool_type", "spatial_output_size",
              "feature_dim", "hidden_dim", "conv_channels", "input_resize",
              "center_crop")
TRAIN_KEYS = ("objective", "optimizer", "learning_rate", "lr_gamma", "kl_weight",
              "epochs", "early_stopping_patience")


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def read_config(run_dir):
    """The config the run actually used, from whichever checkpoint exists.

    Loading a checkpoint costs a second and pulls in torch, so this degrades to
    an empty dict rather than failing the harvest: the curves are the part that
    cannot be reconstructed later, and they need no torch at all.
    """
    for name in ("last.pt", "best.pt"):
        path = run_dir / "checkpoints" / name
        if not path.exists():
            continue
        try:
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            print("  config unreadable from {}: {}".format(name, exc))
            return {}
        config = payload.get("config") or {}
        config["_checkpoint_epoch"] = payload.get("epoch")
        return config
    return {}


def summarise(run, history, config):
    row = {"run": run, "epochs_completed": len(history),
           "git_revision": git_revision()}
    losses = [item.get("train_loss") for item in history if item.get("train_loss") is not None]
    values = [item.get("selection_value") for item in history
              if item.get("selection_value") is not None]
    if losses:
        row["final_train_loss"] = round(losses[-1], 6)
        row["min_train_loss"] = round(min(losses), 6)
        row["last20_train_loss"] = round(sum(losses[-20:]) / len(losses[-20:]), 6)
    if values:
        best = max(range(len(values)), key=lambda i: values[i])
        row["best_val_metric"] = round(values[best], 6)
        row["best_epoch"] = best + 1
        row["mean_val_metric"] = round(sum(values) / len(values), 6)
        row["final_val_metric"] = round(values[-1], 6)
    data = config.get("data", {})
    model = config.get("model", {})
    train = config.get("train", {})
    row["datasets"] = "+".join(data.get("active_datasets", []) or [])
    # Section prefixes, so a config key can never collide with a computed one --
    # train.epochs silently overwrote the completed-epoch count before this.
    for section, keys in (("data", DATA_KEYS), ("model", MODEL_KEYS),
                          ("train", TRAIN_KEYS)):
        source = {"data": data, "model": model, "train": train}[section]
        for key in keys:
            row["{}.{}".format(section, key)] = source.get(key)
    row["model.init_extractor"] = model.get("init_extractor")
    row["model.freeze_extractor"] = model.get("freeze_extractor")
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--output", default=str(ROOT / "results"))
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Only these run directory names; default is every "
                             "run_* directory that has a history.")
    parser.add_argument("--skip-configs", action="store_true",
                        help="Do not open checkpoints. Faster, but the summary "
                             "loses the settings that distinguish the runs.")
    args = parser.parse_args()

    artifacts, output = Path(args.artifacts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    wanted = set(args.runs) if args.runs else None

    curves, summaries = [], []
    for run_dir in sorted(artifacts.glob("run_*")):
        if not run_dir.is_dir() or (wanted and run_dir.name not in wanted):
            continue
        history_path = run_dir / "logs" / "history.json"
        if not history_path.exists():
            continue
        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
        if not history:
            continue
        print("{}  ({} epochs)".format(run_dir.name, len(history)))

        destination = output / run_dir.name
        (destination / "logs").mkdir(parents=True, exist_ok=True)
        for name in ("history.json", "history.csv"):
            source = run_dir / "logs" / name
            if source.exists():
                shutil.copy2(source, destination / "logs" / name)
        reports = sorted((run_dir / "reports").glob("*.json")) \
            if (run_dir / "reports").is_dir() else []
        if reports:
            (destination / "reports").mkdir(parents=True, exist_ok=True)
            for report in reports:
                shutil.copy2(report, destination / "reports" / report.name)
            print("  + {} report(s)".format(len(reports)))

        config = {} if args.skip_configs else read_config(run_dir)
        if config:
            with open(destination / "config.json", "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, default=str)
        summaries.append(summarise(run_dir.name, history, config))
        for item in history:
            validation = item.get("validation") or {}
            curves.append({
                "run": run_dir.name,
                "epoch": item.get("epoch"),
                "train_loss": item.get("train_loss"),
                "learning_rate": item.get("learning_rate"),
                "selection_metric": item.get("selection_metric"),
                "selection_value": item.get("selection_value"),
                "auroc": validation.get("auroc"),
                "eer": validation.get("eer"),
                "tpr_at_target_fpr": validation.get("tpr_at_target_fpr"),
                "embedding_variance_mean": validation.get("embedding_variance_mean"),
            })

    if not summaries:
        print("No run with a history was found under {}.".format(artifacts))
        return 1

    with open(output / "curves.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    fields = []
    for row in summaries:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(output / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    with open(output / "collected.json", "w", encoding="utf-8") as handle:
        json.dump({"collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "git_revision": git_revision(),
                   "runs": [row["run"] for row in summaries],
                   "epoch_rows": len(curves)}, handle, indent=2)

    print("\n{} runs, {} epoch rows".format(len(summaries), len(curves)))
    print("wrote {}, {} and per-run logs under {}".format(
        output / "curves.csv", output / "summary.csv", output))
    print("\nCheckpoints are NOT copied -- they are too large for git. Back up "
          "best.pt for any run worth re-evaluating before the pod is returned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
