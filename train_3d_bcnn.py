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

from video_bcnn.data import load_manifest, seed_worker
from video_bcnn.experiment import (
    active_records,
    capped_validation_records,
    dataset_balanced_sampler,
    evaluate_values,
    make_dataset,
    resolve_objective,
    score_loader,
    select_records,
    training_records,
)
from video_bcnn.model import Stable2DFeatureExtractor, Stable3DFeatureExtractor, VideoBayesianCNN
from video_bcnn.reporting import save_history
from video_bcnn.utils import ensure_dir, load_config, resolve_device, seed_everything


def build_model(config, device):
    expected = {
        "spatial_kernel_size": 5,
        "spatial_pool_kernel_size": 4,
        "spatial_pool_stride": 2,
        "feature_dim": 15488,
        "hidden_dim": 512,
    }
    for name, value in expected.items():
        configured = config["model"].get(name)
        if configured != value:
            raise ValueError(
                "Controlled experiment requires model.{}={}, received {}.".format(
                    name, value, configured
                )
            )
    architecture = config["model"].get("architecture", "3d")
    if architecture == "3d":
        extractor = Stable3DFeatureExtractor(
            temporal_kernel_size=config["model"]["temporal_kernel_size"],
            conv_channels=config["model"]["conv_channels"],
        ).to(device)
    elif architecture == "2d_score_mean":
        extractor = Stable2DFeatureExtractor().to(device)
    else:
        raise ValueError("Unknown architecture: {}".format(architecture))
    model = VideoBayesianCNN(
        extractor,
        dropout=config["model"]["dropout"],
        prior_std=config["model"]["prior_std"],
        observation_std=config["model"]["observation_std"],
        rho_init=config["model"]["posterior_rho_init"],
        kl_weight=config["train"]["kl_weight"],
        likelihood=config["train"].get("observation_likelihood", "gaussian"),
    )
    return extractor, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
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
    scheduler = pyro.optim.ExponentialLR({
        "optimizer": torch.optim.SGD,
        "optim_args": {
            "lr": float(config["train"]["learning_rate"]),
            "momentum": float(config["train"]["momentum"]),
        },
        "gamma": float(config["train"]["lr_gamma"]),
    }, clip_args={"clip_norm": float(config["train"]["gradient_clip_norm"])})
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
        for batch in progress:
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
