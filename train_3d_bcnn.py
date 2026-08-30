"""Train the controlled 3D-CNN + Bayesian FC model on real clips only."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import SVI, TraceGraph_ELBO
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_bcnn.data import (
    CachedClipDataset,
    load_manifest,
    seed_worker,
    skip_unreadable_collate,
)
from video_bcnn.experiment import (
    active_records,
    capped_validation_records,
    dataset_balanced_sampler,
    evaluate_values,
    make_dataset,
    overfit_records,
    resolve_objective,
    subset_training_identities,
    score_loader,
    select_records,
    training_records,
)
from video_bcnn.model import (
    ACTIVATIONS,
    Stable2DFeatureExtractor,
    Stable3DFeatureExtractor,
    VideoBayesianCNN,
    feature_dimension,
)
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


def build_model(config, device):
    # These are hard-coded inside the extractors, so a config that disagrees
    # would describe a model that was never built.
    structural = {
        "spatial_kernel_size": 5,
        "spatial_pool_kernel_size": 4,
        "spatial_pool_stride": 2,
    }
    for name, value in structural.items():
        configured = config["model"].get(name)
        if configured != value:
            raise ValueError(
                "This extractor implements model.{}={}, received {}.".format(
                    name, value, configured
                )
            )
    settings = {
        "conv_channels": config["model"]["conv_channels"],
        "activation": config["model"].get("activation", "sigmoid"),
        "spatial_output_size": config["model"].get("spatial_output_size", 22),
        "pool_type": config["model"].get("spatial_pool_type", "avg"),
        "norm": config["model"].get("norm", "none"),
    }
    architecture = config["model"].get("architecture", "3d")
    if architecture == "3d":
        extractor = Stable3DFeatureExtractor(
            temporal_kernel_size=config["model"]["temporal_kernel_size"], **settings
        ).to(device)
    elif architecture == "2d_score_mean":
        extractor = Stable2DFeatureExtractor(**settings).to(device)
    else:
        raise ValueError("Unknown architecture: {}".format(architecture))
    # feature_dim follows from conv_channels and spatial_output_size. It stays
    # in the config as a recorded protocol value, but a stale entry must not
    # silently disagree with the model that actually ran.
    declared = config["model"].get("feature_dim")
    if declared is not None and int(declared) != int(extractor.feature_dim):
        raise ValueError(
            "model.feature_dim={} contradicts the extractor's {} "
            "(conv_channels={}, spatial_output_size={}). Update the config.".format(
                declared, extractor.feature_dim,
                settings["conv_channels"], settings["spatial_output_size"],
            )
        )
    model = VideoBayesianCNN(
        extractor,
        dropout=config["model"]["dropout"],
        prior_std=config["model"]["prior_std"],
        observation_std=config["model"]["observation_std"],
        rho_init=config["model"]["posterior_rho_init"],
        kl_weight=config["train"]["kl_weight"],
        likelihood=config["train"].get("observation_likelihood", "gaussian"),
        hidden_dim=config["model"].get("hidden_dim", 512),
    )
    return extractor, model


def apply_sweep_overrides(config, args):
    """Fold command-line architecture/optimiser overrides into the config.

    They are written into the config rather than applied later so the
    checkpoint records the settings the run actually used.
    """
    if args.activation is not None:
        config["model"]["activation"] = args.activation
    if args.spatial_output_size is not None:
        config["model"]["spatial_output_size"] = int(args.spatial_output_size)
    if args.conv_channels is not None:
        config["model"]["conv_channels"] = [int(value) for value in args.conv_channels]
    if args.pool_type is not None:
        config["model"]["spatial_pool_type"] = args.pool_type
    if args.norm is not None:
        config["model"]["norm"] = args.norm
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = int(args.hidden_dim)
    if args.spatial_output_size is not None or args.conv_channels is not None:
        # feature_dim is derived, so keep the recorded value honest.
        config["model"]["feature_dim"] = feature_dimension(
            config["model"]["conv_channels"][-1],
            config["model"].get("spatial_output_size", 22),
        )
    if args.optimizer is not None:
        config["train"]["optimizer"] = args.optimizer
    if args.kl_weight is not None:
        config["train"]["kl_weight"] = float(args.kl_weight)
    if args.prior_std is not None:
        config["model"]["prior_std"] = float(args.prior_std)
    if args.posterior_rho_init is not None:
        config["model"]["posterior_rho_init"] = float(args.posterior_rho_init)
    if args.learning_rate is not None:
        config["train"]["learning_rate"] = float(args.learning_rate)
    if args.selection_max_fakes is not None:
        config["data"]["selection_max_fakes_per_dataset"] = int(args.selection_max_fakes)
    if args.early_stopping_patience is not None:
        config["train"]["early_stopping_patience"] = int(args.early_stopping_patience)
    if args.clip_length is not None:
        config["data"]["clip_length"] = int(args.clip_length)
    if args.train_clip_strides is not None:
        config["data"]["train_clip_strides"] = [int(value) for value in args.train_clip_strides]
    if args.eval_clip_stride is not None:
        config["data"]["eval_clip_stride"] = int(args.eval_clip_stride)
    if args.run_suffix:
        for key in ("checkpoint_dir", "log_dir", "report_dir"):
            path = Path(config["train"][key])
            config["train"][key] = str(
                path.parent.parent / (path.parent.name + "_" + args.run_suffix) / path.name
            )
    return config


def build_optimizer(config):
    """SGD reproduces the paper; Adam exists because SGD moved the ELBO 0.2%."""
    name = str(config["train"].get("optimizer", "sgd")).lower()
    learning_rate = float(config["train"]["learning_rate"])
    if name == "sgd":
        optimizer, arguments = torch.optim.SGD, {
            "lr": learning_rate, "momentum": float(config["train"]["momentum"]),
        }
    elif name == "adam":
        optimizer, arguments = torch.optim.Adam, {
            "lr": learning_rate,
            "betas": tuple(config["train"].get("adam_betas", (0.9, 0.999))),
        }
    else:
        raise ValueError("Unknown train.optimizer {!r}; expected sgd or adam.".format(name))
    return pyro.optim.ExponentialLR({
        "optimizer": optimizer,
        "optim_args": arguments,
        "gamma": float(config["train"]["lr_gamma"]),
    }, clip_args={"clip_norm": float(config["train"]["gradient_clip_norm"])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    # Architecture/optimiser sweeps: overriding on the command line keeps one
    # config per experiment instead of one per hypothesis, and the values still
    # land in the checkpoint because they are written into the config.
    parser.add_argument("--activation", default=None,
                        choices=sorted(ACTIVATIONS),
                        help="Override model.activation (default: the config's, sigmoid).")
    parser.add_argument("--spatial-output-size", type=int, default=None,
                        help="Override model.spatial_output_size. 22 keeps feature_dim=15488; "
                             "7 gives 1568 and 4 gives 512, shrinking Bayesian FC1 without "
                             "changing the head's structure.")
    parser.add_argument("--conv-channels", type=int, nargs=3, default=None,
                        metavar=("C1", "C2", "C3"),
                        help="Override model.conv_channels. The paper's 16 24 32 gives a "
                             "90k-parameter extractor against a 7.9M-parameter Bayesian "
                             "head, so 99% of the model is the classifier.")
    parser.add_argument("--pool-type", default=None, choices=["avg", "max"],
                        help="Override model.spatial_pool_type. Three 4x4 average pools "
                             "low-pass exactly the frequencies deepfake artefacts live in.")
    parser.add_argument("--norm", default=None, choices=["none", "batch"],
                        help="Override model.norm: normalisation before each activation. "
                             "Without it the pre-activation magnitude drifted 0.35 -> 8.4 "
                             "across stages.")
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Override model.hidden_dim (Bayesian FC1 output width).")
    parser.add_argument("--optimizer", default=None, choices=["sgd", "adam"],
                        help="Override train.optimizer.")
    parser.add_argument("--kl-weight", type=float, default=None,
                        help="Override train.kl_weight. The KL is scaled by "
                             "kl_weight/clips_per_epoch, and with 7.9M Bayesian FC1 "
                             "weights it otherwise dwarfs the likelihood.")
    parser.add_argument("--prior-std", type=float, default=None,
                        help="Override model.prior_std.")
    parser.add_argument("--posterior-rho-init", type=float, default=None,
                        help="Override model.posterior_rho_init. Less negative means a "
                             "wider initial posterior and a much smaller KL.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override train.learning_rate.")
    parser.add_argument("--selection-max-fakes", type=int, default=None,
                        help="Validation fakes per dataset used for model selection. "
                             "The default 64 makes AUROC swing +/-0.04 between epochs, "
                             "which is wider than the differences between runs.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Override train.early_stopping_patience.")
    # CelebDFv3 runs at 28 fps, so the default 8 frames at stride 1 span 0.28 s.
    # Temporal forgery artefacts (identity drift, blink rhythm, expression
    # dynamics) live on a 1-3 s scale, which no amount of extractor capacity
    # can recover from a window that short.
    parser.add_argument("--clip-length", type=int, default=None,
                        help="Frames per clip. Combined with the strides this "
                             "sets how many seconds the 3D convolutions see.")
    parser.add_argument("--train-clip-strides", type=int, nargs="+", default=None,
                        help="Frame strides sampled during training, e.g. 4 8.")
    parser.add_argument("--eval-clip-stride", type=int, default=None,
                        help="Frame stride for validation and test clips.")
    parser.add_argument("--run-suffix", default=None,
                        help="Append to the run directory name so sweep variants "
                             "do not overwrite each other.")
    parser.add_argument(
        "--train-identities",
        type=int,
        default=None,
        metavar="N",
        help="Train on N identities per dataset instead of all of them, keeping "
             "the donor-safe rule. Isolates identity diversity from the video "
             "count, resolution and forgery-method differences that separate "
             "CelebDFv3 from DFD.",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=None,
        metavar="N",
        help="Train on N videos per dataset per class but score the real held-out "
             "val split. Unlike --overfit-subset this measures generalisation, "
             "which is the question the overfit diagnostic cannot answer.",
    )
    parser.add_argument(
        "--cache-clips",
        action="store_true",
        help="Keep decoded clips in RAM: pins clip positions, disables the flip, "
             "and forces num_workers=0. Trades augmentation for speed, so it is a "
             "diagnostic setting, not a protocol one.",
    )
    parser.add_argument(
        "--overfit-subset",
        type=int,
        default=None,
        metavar="N",
        help="Diagnostic: train on N videos per dataset per class and score the "
             "same videos. Augmentation off, clip positions pinned, clips cached "
             "in RAM. A model that cannot fit this has an architecture or "
             "optimisation problem, and the answer takes minutes.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Repoint a dataset root for this machine, e.g. "
             "--dataset-root DFD=/root/autodl-tmp/DFD-Kaggle. Repeatable. "
             "Overrides the DFD_ROOT / CELEBDFV3_ROOT environment variables.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers for this machine. Defaults to $NUM_WORKERS, then "
             "the config. Measure with scripts/benchmark_loader.py -- more is not "
             "automatically faster.",
    )
    args = parser.parse_args()
    config = override_dataset_roots(load_config(args.config), args.dataset_root)
    override_num_workers(config, args.num_workers)
    apply_sweep_overrides(config, args)
    print("Dataset roots: {}".format(verify_dataset_roots(config)))
    if args.max_epochs is not None:
        config["train"]["epochs"] = int(args.max_epochs)
    seed_everything(config["seed"])
    pyro.clear_param_store()
    device = resolve_device(config["device"])

    objective = resolve_objective(config)
    records = active_records(load_manifest(args.manifest), config)
    train_records = training_records(records, objective)
    validation_candidates = select_records(records, "val")
    validation_records = capped_validation_records(
        validation_candidates,
        config["seed"],
        config["data"].get("selection_max_fakes_per_dataset"),
    )
    if not train_records or not validation_records:
        raise ValueError("The manifest lacks training or real/fake validation videos.")
    if args.overfit_subset and args.train_subset:
        raise ValueError("--overfit-subset and --train-subset are mutually exclusive.")
    if args.cache_clips or args.overfit_subset:
        # A cached clip is decoded once, so its position and flip must be fixed.
        config["data"].update({
            "horizontal_flip_probability": 0.0,
            "deterministic_train_clips": True,
            "num_workers": 0,
        })
    if args.train_identities:
        train_records, identity_summary = subset_training_identities(
            train_records, args.train_identities, config["seed"]
        )
        config["data"]["train_identities"] = int(args.train_identities)
        print("IDENTITY SUBSET: {}".format(identity_summary))
    if args.train_subset:
        train_records = overfit_records(train_records, args.train_subset, config["seed"])
        config["data"].update({
            "train_samples_per_dataset_per_epoch": int(args.train_subset),
            "train_balance_keys": ["dataset", "class_name"],
            "train_stratify_key": None,
        })
        for key in ("checkpoint_dir", "log_dir", "report_dir"):
            path = Path(config["train"][key])
            config["train"][key] = str(
                path.parent.parent / (path.parent.name + "_subset") / path.name
            )
        print(
            "SUBSET RUN: {} training videos per dataset per class, scored on the "
            "real held-out val split. These numbers measure generalisation.".format(
                args.train_subset
            )
        )
    if args.overfit_subset:
        train_records = overfit_records(train_records, args.overfit_subset, config["seed"])
        # Scoring the training videos themselves is the point: this measures
        # capacity, not generalisation.
        validation_records = list(train_records)
        config["data"].update({
            "selection_clips_per_video": 1,
            "train_samples_per_dataset_per_epoch": int(args.overfit_subset),
            "train_balance_keys": ["dataset", "class_name"],
            "train_stratify_key": None,
        })
        config["train"].update({
            "early_stopping_patience": int(config["train"]["epochs"]) + 1,
            "selection_min_metric_improvement": 0.0,
        })
        for key in ("checkpoint_dir", "log_dir", "report_dir"):
            path = Path(config["train"][key])
            config["train"][key] = str(
                path.parent.parent / (path.parent.name + "_overfit") / path.name
            )
        print(
            "OVERFIT DIAGNOSTIC: {} videos per dataset per class, scored on the "
            "training videos themselves. These numbers measure capacity, not "
            "detection performance.".format(args.overfit_subset)
        )
    # Printed here because the overfit block above forces workers to 0.
    print("DataLoader workers: {}".format(config["data"]["num_workers"]))
    active_datasets = list(config["data"].get("active_datasets", []))
    train_counts = {
        dataset: {
            "real": sum(
                row["dataset"] == dataset and int(row["label"]) == 1 for row in train_records
            ),
            "fake": sum(
                row["dataset"] == dataset and int(row["label"]) == 0 for row in train_records
            ),
        }
        for dataset in sorted({row["dataset"] for row in train_records})
    }
    if objective["name"] == "supervised":
        without_fakes = [
            dataset for dataset, counts in train_counts.items() if counts["fake"] == 0
        ]
        if without_fakes:
            raise ValueError(
                "Supervised training needs labelled fake training videos, but {} have none. "
                "Build the supervised manifest first: "
                "python scripts/build_supervised_manifest.py".format(without_fakes)
            )
    validation_counts = {
        dataset: {
            "real": sum(
                row["dataset"] == dataset and int(row["label"]) == 1
                for row in validation_records
            ),
            "fake": sum(
                row["dataset"] == dataset and int(row["label"]) == 0
                for row in validation_records
            ),
        }
        for dataset in active_datasets
    }
    missing_train = [dataset for dataset in active_datasets if train_counts.get(dataset, 0) == 0]
    incomplete_validation = [
        dataset for dataset, counts in validation_counts.items()
        if counts["real"] == 0 or counts["fake"] == 0
    ]
    if missing_train or incomplete_validation:
        raise ValueError(
            "Invalid combined protocol: missing real training datasets={}, "
            "validation datasets without both classes={}.".format(
                missing_train, incomplete_validation
            )
        )
    print("Using {}. Objective: {}. Training videos: {}".format(
        device, objective["name"], train_counts
    ))
    print(
        "Validation videos by dataset: {} (model selection uses macro dataset AUROC).".format(
            validation_counts
        )
    )
    if objective["name"] == "supervised":
        print(
            "Protocol note: supervised training uses labelled fakes whose target and donor "
            "identities are both training identities. Cross-dataset AUROC is the "
            "generalization number; same-dataset AUROC is an upper bound."
        )
    else:
        print("Protocol note: labeled validation fakes are used for debug-stage checkpoint selection.")

    train_dataset = make_dataset(train_records, config, training=True)
    validation_dataset = make_dataset(
        validation_records,
        config,
        training=False,
        clips_per_video=config["data"]["selection_clips_per_video"],
    )
    if args.overfit_subset or args.cache_clips:
        # Decode once, then every later epoch is GPU-bound.
        train_dataset = CachedClipDataset(train_dataset)
        validation_dataset = CachedClipDataset(validation_dataset)
    default_balance_keys = ["dataset", "class_name"] if objective["name"] == "supervised" else ["dataset"]
    balance_keys = list(config["data"].get("train_balance_keys", default_balance_keys))
    sampler = dataset_balanced_sampler(
        train_records,
        config["seed"],
        config["data"].get("train_samples_per_dataset_per_epoch", "max"),
        group_keys=balance_keys,
        stratify_key=config["data"].get("train_stratify_key"),
    )
    loader_options = {
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        # A corrupt file is dropped and counted instead of ending the run.
        "collate_fn": skip_unreadable_collate,
    }
    batch_size = int(config["train"]["physical_batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options
    )
    extractor, model = build_model(config, device)
    print(
        "Extractor: {} activation, {} pooling, {} norm, conv_channels={}, {}x{} spatial "
        "-> feature_dim={} ({:,} parameters). Bayesian head: {} -> {} -> 1 ({:,} FC1 "
        "weights). Optimizer: {}.".format(
            config["model"].get("activation", "sigmoid"),
            config["model"].get("spatial_pool_type", "avg"),
            config["model"].get("norm", "none"),
            list(extractor.conv_channels),
            extractor.spatial_output_size, extractor.spatial_output_size,
            extractor.feature_dim,
            sum(parameter.numel() for parameter in extractor.parameters()),
            extractor.feature_dim, model.hidden_dim,
            extractor.feature_dim * model.hidden_dim,
            config["train"].get("optimizer", "sgd"),
        )
    )
    scheduler = build_optimizer(config)
    svi = SVI(model.model, model.guide, scheduler, loss=TraceGraph_ELBO())

    checkpoint_dir = ensure_dir(config["train"]["checkpoint_dir"])
    log_dir = ensure_dir(config["train"]["log_dir"])
    best_value, best_epoch, patience, history = -float("inf"), 0, 0, []
    current_lr = float(config["train"]["learning_rate"])
    num_train_clips = len(sampler)
    print(
        "Balanced epoch: {} clips ({} per group, groups balanced on {}{}).".format(
            len(sampler), sampler.samples_per_group, balance_keys,
            ", stratified by {}".format(sampler.stratify_key) if sampler.stratify_key else "",
        )
    )
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        extractor.train()
        total_loss, batches = 0.0, 0
        progress = tqdm(train_loader, desc="3D BCNN epoch {}/{}".format(epoch, config["train"]["epochs"]))
        skipped_clips = 0
        for batch in progress:
            if batch is None:
                skipped_clips += 1
                continue
            clips = batch["clip"].to(device, non_blocking=True)
            if extractor.input_mode == "frame":
                batch_size, channels, frame_count, height, width = clips.shape
                model_inputs = clips.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * frame_count, channels, height, width
                )
                units_per_observation = frame_count
            else:
                model_inputs = clips
                units_per_observation = 1
            if objective["constant_target"] is None:
                # Supervised: regress/classify the manifest label (real=1, fake=0).
                targets = batch["label"].to(device=device, dtype=torch.float32)
            else:
                targets = torch.full(
                    (clips.shape[0],), float(objective["constant_target"]),
                    device=device, dtype=torch.float32,
                )
            loss = float(svi.step(
                model_inputs, targets, num_train_clips, units_per_observation
            ))
            if not np.isfinite(loss):
                raise FloatingPointError("Non-finite negative ELBO in epoch {}.".format(epoch))
            total_loss += loss
            batches += 1
            progress.set_postfix(elbo="{:.4f}".format(total_loss / batches))
        if skipped_clips:
            print("Skipped {} unreadable training clips this epoch.".format(skipped_clips))
        if epoch % int(config["train"]["lr_step_interval"]) == 0:
            scheduler.step()
            current_lr *= float(config["train"]["lr_gamma"])

        values = score_loader(
            model,
            validation_loader,
            device,
            config["train"].get("report_mc_uncertainty_samples", 0),
            score_sign=objective["score_sign"],
        )
        validation, threshold = evaluate_values(
            values, config["train"]["calibration_false_positive_rate"]
        )
        metric_name = config["train"].get("selection_metric", "macro_dataset_auroc")
        selection_value = float(validation[metric_name])
        diagnostics = model.diagnostics()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            "skipped_training_clips": int(skipped_clips),
            "learning_rate": current_lr,
            "selection_metric": metric_name,
            "selection_value": selection_value,
            "validation": validation,
            "posterior_diagnostics": diagnostics,
        }
        history.append(row)
        dataset_auc = {name: item["auroc"] for name, item in validation["per_dataset"].items()}
        print(
            "Validation AUROC={:.4f}, macro={:.4f}, TPR@{:.0%}FPR={:.4f}, per-dataset={}, embedding-var={:.6g}".format(
                validation["auroc"], validation["macro_dataset_auroc"],
                validation["target_false_positive_rate"], validation["tpr_at_target_fpr"],
                dataset_auc, validation["embedding_variance_mean"],
            )
        )
        crop_audit = validation["preprocessing_diagnostics"]["overall"]
        print(
            "Preprocessing: clip-any-miss={:.2%}, frame-miss={:.2%}, FPS={:.2f}+/-{:.2f}".format(
                crop_audit["clip_any_face_detection_failure_rate"],
                crop_audit["frame_face_detection_failure_fraction"],
                crop_audit["fps_mean"], crop_audit["fps_std"],
            )
        )
        checkpoint = {
            "architecture": config["model"].get("architecture", "3d"),
            "epoch": epoch,
            "feature_extractor": extractor.state_dict(),
            "pyro_params": pyro.get_param_store().get_state(),
            "config": config,
            "threshold": threshold,
            "validation": validation,
            "diagnostics": diagnostics,
            "selection_protocol": "labeled_fake_debug_macro_dataset_auroc",
            "objective": objective,
            "score_sign": float(objective["score_sign"]),
            "data_protocol": {
                "active_datasets": active_datasets,
                "objective": objective["name"],
                "training_videos": train_counts,
                "balance_keys": balance_keys,
                "stratify_key": sampler.stratify_key,
                "balanced_training_clips_per_epoch": num_train_clips,
                "selection_validation_videos": validation_counts,
            },
        }
        minimum = float(config["train"].get("selection_min_metric_improvement", 0.0))
        improved = best_epoch == 0 or selection_value >= best_value + minimum
        if improved:
            best_value, best_epoch, patience = selection_value, epoch, 0
            torch.save(checkpoint, checkpoint_dir / "best.pt")
        else:
            patience += 1
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        save_history(history, log_dir)
        if patience >= int(config["train"]["early_stopping_patience"]):
            print("Early stopping. Best epoch: {} ({}={:.4f}).".format(best_epoch, metric_name, best_value))
            break
    print("Best checkpoint: {}".format((checkpoint_dir / "best.pt").resolve()))


if __name__ == "__main__":
    main()
