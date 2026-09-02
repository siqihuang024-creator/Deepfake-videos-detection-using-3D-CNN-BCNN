"""Pre-train the 3D feature extractor behind a deterministic binary head.

Stage A of the two-stage protocol. The extractor is trained on real and fake
clips with ordinary cross-entropy, then its weights are saved and the head is
thrown away; `train_3d_bcnn.py --init-extractor` loads them and attaches the
Bayesian head for stage B.

The point is to keep the posterior out of the extractor's gradients. Trained
jointly, every step the extractor takes is computed from sampled weights, and
the KL term was 95-99.9% of the objective for the first five runs; even after
the weight was cut to 6e-6 the best joint run reached a training loss of only
0.6576, barely under ln 2. A deterministic head removes both effects.

Restrict the fakes to one resolution family (--fake-methods) unless the shortcut
described in that flag's help has been dealt with some other way: an extractor
that learns "blurry means fake" gives stage B a feature space organised by
resolution, and every real video in CelebDFv3 is native wide.
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_bcnn.data import load_manifest, seed_worker, skip_unreadable_collate
from video_bcnn.experiment import (
    active_records,
    capped_validation_records,
    dataset_balanced_sampler,
    filter_forgery_methods,
    make_dataset,
    select_records,
    subset_training_identities,
    training_records,
    resolve_objective,
)
from video_bcnn.metrics import detection_metrics, calibrate_threshold
from video_bcnn.model import DeterministicHead
from video_bcnn.reporting import save_history
from video_bcnn.utils import (
    ensure_dir,
    load_config,
    override_dataset_roots,
    override_num_workers,
    resolve_device,
    seed_everything,
    verify_dataset_roots,
)
from train_3d_bcnn import build_model, apply_sweep_overrides


def json_safe(value):
    """Replace non-finite floats with None so save_json can write the history.

    save_json uses allow_nan=False. A degenerate validation split -- which a
    truncated smoke test produces -- yields NaN metrics that would otherwise
    end the run at the point it writes its first epoch.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_loader(dataset, config, sampler=None, shuffle=False, batch_size=1):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        shuffle=shuffle,
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=skip_unreadable_collate,
        worker_init_fn=seed_worker,
        drop_last=False,
    )


@torch.no_grad()
def score_split(extractor, head, loader, device, limit=None):
    """One logit per video, averaged over its evaluation clips."""
    extractor.eval()
    head.eval()
    labels, scores, skipped = [], [], 0
    for index, batch in enumerate(tqdm(loader, desc="Scoring", leave=False)):
        if limit is not None and index >= limit:
            break
        if batch is None:
            skipped += 1
            continue
        clips = batch["clips"].to(device)
        shape = clips.shape
        # [batch, clips, C, T, H, W] -> one row per clip
        flat = clips.reshape(-1, *shape[-4:])
        logits = head(extractor(flat)).reshape(shape[0], -1).mean(dim=1)
        labels.extend(int(value) for value in batch["label"])
        scores.extend(float(value) for value in logits.detach().cpu())
    return np.asarray(labels), np.asarray(scores), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--run-suffix", default="pretrain")
    parser.add_argument("--dataset-root", action="append", default=None, metavar="NAME=PATH")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--selection-max-fakes", type=int, default=None)
    parser.add_argument(
        "--fake-methods", nargs="+", default=None, metavar="SUBSTRING",
        help="Train on fakes whose method contains one of these substrings; "
             "reals are always kept. Pass 'FaceSwap/' plus 'LivePortrait' for "
             "CelebDFv3's resolution-matched subset, so the extractor cannot "
             "learn that blur means fake.",
    )
    parser.add_argument("--train-identities", type=int, default=None, metavar="N")
    # The architecture flags mirror train_3d_bcnn.py so stage A and stage B can
    # be given identical extractor settings; a mismatch would make the saved
    # weights unloadable.
    parser.add_argument("--activation", default=None, choices=sorted(("sigmoid", "relu", "leaky_relu", "tanh")))
    parser.add_argument("--conv-channels", nargs=3, type=int, default=None)
    parser.add_argument("--spatial-output-size", type=int, default=None)
    parser.add_argument("--pool-type", default=None, choices=["avg", "max"])
    parser.add_argument("--norm", default=None, choices=["none", "batch"])
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--active-datasets", nargs="+", default=None, metavar="NAME")
    parser.add_argument(
        "--frame-mode", choices=["face", "letterbox"], default=None,
        help="'face' crops the tracked box as every run up to v15 did. "
             "'letterbox' keeps the whole frame, rescaled to fit and padded, "
             "and skips face detection entirely.",
    )
    parser.add_argument(
        "--letterbox-size", nargs=2, type=int, default=None, metavar=("W", "H"),
        help="Canvas for --frame-mode letterbox, e.g. 896 504. At 896x504 "
             "CelebDFv3 sits at its native scale (median factor 1.01) and "
             "DFD is reduced by 0.47, while the whole scene stays in view.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Clips per optimiser step. It was pinned at 1 while Haar made "
             "decoding the bottleneck; without detection a larger batch is "
             "affordable and cuts the gradient noise that a single-clip BCE "
             "step carries.",
    )
    parser.add_argument(
        "--smoke-test", type=int, default=None, metavar="N",
        help="Run N training batches and a truncated validation, then exit. "
             "Walks every code path in minutes before a long run.",
    )
    parser.add_argument(
        "--samples-per-group", type=int, default=None, metavar="N",
        help="Clips drawn per balance group per epoch; every training video "
             "stays in the pool. The config default 'max' matches the largest "
             "group, which oversamples the 613 CelebDFv3 reals up to the 30k "
             "fakes and makes one epoch 120k clips. Pass a budget instead and "
             "run more epochs. Unlike train_3d_bcnn.py's --train-subset this "
             "does not drop any training video.",
    )
    parser.add_argument("--crop-padding", choices=["clamp", "replicate"], default=None)
    parser.add_argument("--face-crop", choices=["on", "off"], default=None)
    parser.add_argument("--face-margin", type=float, default=None)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--train-clip-strides", type=int, nargs="+", default=None)
    parser.add_argument("--eval-clip-stride", type=int, default=None)
    args = parser.parse_args()

    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    override_num_workers(config, args.num_workers)
    # apply_sweep_overrides reads flags this script does not define, so fill in
    # the ones it expects with the neutral value. run_suffix is held aside
    # rather than read back from args: the helper renames the run directory for
    # a Bayesian run, and this script names its own below.
    run_suffix = args.run_suffix
    for name in ("optimizer", "kl_weight", "prior_std", "posterior_rho_init",
                 "early_stopping_patience", "objective", "mc_uncertainty_samples",
                 "run_suffix"):
        setattr(args, name, None)
    apply_sweep_overrides(config, args)
    if args.active_datasets is not None:
        config["data"]["active_datasets"] = list(args.active_datasets)
    if args.frame_mode is not None:
        config["data"]["frame_mode"] = args.frame_mode
    if args.letterbox_size is not None:
        config["data"]["letterbox_size"] = [int(v) for v in args.letterbox_size]
    if args.crop_padding is not None:
        config["data"]["crop_padding"] = args.crop_padding
    if args.selection_max_fakes is not None:
        config["data"]["selection_max_fakes_per_dataset"] = int(args.selection_max_fakes)
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = float(args.learning_rate)
    config["train"]["epochs"] = int(args.max_epochs)
    config["train"]["stage"] = "pretrain"
    for key in ("checkpoint_dir", "log_dir", "report_dir"):
        path = Path(config["train"][key])
        config["train"][key] = str(
            path.parent.parent / (path.parent.name + "_" + run_suffix) / path.name
        )

    verify_dataset_roots(config)
    device = resolve_device(config["device"])
    seed_everything(config["seed"])
    print("Dataset roots: {} | DataLoader workers: {}".format(
        {name: str(root) for name, root in config["data"]["dataset_roots"].items()},
        config["data"].get("num_workers", 0)))

    records = active_records(load_manifest(args.manifest), config)
    objective = resolve_objective(dict(config, train=dict(config["train"], objective="supervised")))
    train_records = training_records(records, objective)
    validation_candidates = select_records(records, "val")
    if args.fake_methods:
        train_records, dropped = filter_forgery_methods(train_records, args.fake_methods)
        validation_candidates, _ = filter_forgery_methods(validation_candidates, args.fake_methods)
        config["data"]["fake_methods"] = list(args.fake_methods)
        print("METHOD FILTER: keeping {}; dropped {} fake training videos across "
              "{} methods.".format(list(args.fake_methods), sum(dropped.values()), len(dropped)))
    if args.train_identities:
        train_records, summary = subset_training_identities(
            train_records, args.train_identities, config["seed"])
        config["data"]["train_identities"] = int(args.train_identities)
        print("IDENTITY SUBSET: {}".format(summary))
    validation_records = capped_validation_records(
        validation_candidates, config["seed"],
        config["data"].get("selection_max_fakes_per_dataset"))

    train_dataset = make_dataset(train_records, config, training=True)
    validation_dataset = make_dataset(
        validation_records, config, training=False,
        clips_per_video=config["data"]["selection_clips_per_video"])
    if args.samples_per_group is not None:
        config["data"]["train_samples_per_dataset_per_epoch"] = int(args.samples_per_group)
    sampler = dataset_balanced_sampler(
        train_records,
        config["seed"],
        config["data"].get("train_samples_per_dataset_per_epoch", "max"),
        group_keys=list(config["data"].get(
            "train_balance_keys", ["dataset", "class_name"])),
        stratify_key=config["data"].get("train_stratify_key"),
    )
    print("Balanced epoch: {} clips ({} per group, groups balanced on {}).".format(
        len(sampler), sampler.samples_per_group,
        list(config["data"].get("train_balance_keys", ["dataset", "class_name"]))))
    train_loader = build_loader(
        train_dataset, config, sampler=sampler, batch_size=args.batch_size)
    validation_loader = build_loader(validation_dataset, config)

    extractor, _ = build_model(config, device)
    head = DeterministicHead(
        extractor.feature_dim,
        hidden_dim=config["model"].get("hidden_dim", 512),
        dropout=config["model"]["dropout"],
    ).to(device)
    parameters = list(extractor.parameters()) + list(head.parameters())
    print("STAGE A pre-training: extractor {:,} parameters, deterministic head "
          "{} -> {} -> 1 ({:,} parameters). Optimizer: adam lr={}.".format(
              sum(p.numel() for p in extractor.parameters()),
              extractor.feature_dim, head.fc1.out_features,
              sum(p.numel() for p in head.parameters()),
              config["train"]["learning_rate"]))
    counts = {}
    for row in train_records:
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    print("Training videos: {} | validation videos: {}".format(counts, len(validation_records)))

    optimizer = torch.optim.Adam(parameters, lr=float(config["train"]["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=float(config["train"].get("lr_gamma", 0.95)))
    criterion = nn.BCEWithLogitsLoss()
    clip_norm = float(config["train"].get("gradient_clip_norm", 5.0))

    checkpoint_dir = ensure_dir(config["train"]["checkpoint_dir"])
    log_dir = ensure_dir(config["train"]["log_dir"])
    history, best_auroc = [], -float("inf")

    for epoch in range(1, int(args.max_epochs) + 1):
        extractor.train()
        head.train()
        running, seen, skipped = 0.0, 0, 0
        progress = tqdm(train_loader, desc="Stage A epoch {}/{}".format(epoch, args.max_epochs))
        for step, batch in enumerate(progress):
            if args.smoke_test is not None and step >= args.smoke_test:
                break
            if batch is None:
                skipped += 1
                continue
            # The dataset yields one random clip per video while training and a
            # stack of deterministic clips while scoring, under different keys.
            clips = batch["clip"].to(device)
            targets = batch["label"].to(device).float()
            logits = head(extractor(clips))
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
            optimizer.step()
            running += float(loss.detach()) * targets.numel()
            seen += targets.numel()
            progress.set_postfix(loss="{:.4f}".format(running / max(seen, 1)))
        scheduler.step()
        train_loss = running / max(seen, 1)

        labels, scores, eval_skipped = score_split(
            extractor, head, validation_loader, device,
            limit=(args.smoke_test * 4 if args.smoke_test else None))
        # A high logit means "real" here, so the anomaly score is its negation
        # and the metric helper sees the same orientation as stage B.
        anomaly = -scores
        threshold = calibrate_threshold(
            anomaly[labels == 1], config["train"]["calibration_false_positive_rate"])
        metrics = detection_metrics(labels, anomaly, threshold)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "skipped_training_clips": skipped,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "selection_metric": "auroc",
            "selection_value": metrics["auroc"],
            # save_json writes with allow_nan=False, and this stage has no
            # embedding statistics to report, so the field stays null.
            "validation": dict(metrics, embedding_variance_mean=None,
                               per_dataset={}, macro_dataset_auroc=metrics["auroc"]),
        }
        history.append(json_safe(record))
        print("Validation AUROC={:.4f}  EER={:.4f}  TPR@5%FPR={:.4f}  "
              "train BCE={:.4f}  skipped={}".format(
                  metrics["auroc"], metrics["eer"], metrics["tpr_at_target_fpr"],
                  train_loss, skipped + eval_skipped))

        payload = {
            "extractor": extractor.state_dict(),
            "head": head.state_dict(),
            "config": copy.deepcopy(config),
            "epoch": epoch,
            "validation_auroc": metrics["auroc"],
        }
        torch.save(payload, checkpoint_dir / "last.pt")
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            torch.save(payload, checkpoint_dir / "best.pt")
        save_history(history, log_dir)
        if args.smoke_test is not None:
            print("SMOKE TEST PASSED: {} training batches, {} validation "
                  "batches, checkpoint and history written to {}.".format(
                      args.smoke_test, args.smoke_test * 4, log_dir))
            return 0

    print("Best validation AUROC {:.4f}. Extractor weights: {}".format(
        best_auroc, checkpoint_dir / "best.pt"))
    print("Stage B: train_3d_bcnn.py --init-extractor {} --freeze-extractor "
          "--objective one_class_real".format(checkpoint_dir / "best.pt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
