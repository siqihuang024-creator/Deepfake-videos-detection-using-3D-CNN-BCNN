"""Evaluate a trained 3D-CNN + Bayesian FC checkpoint on val/test videos."""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pyro
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_bcnn.data import load_manifest, seed_worker
from video_bcnn.experiment import (
    active_records,
    capped_validation_records,
    evaluate_values,
    make_dataset,
    score_loader,
    select_records,
)
from video_bcnn.reporting import save_evaluation_report
from video_bcnn.utils import ensure_dir, load_checkpoint, load_config, resolve_device, seed_everything
from train_3d_bcnn import build_model


def checkpoint_evaluation_config(saved_config, runtime_config, eval_datasets=None):
    """Keep trained preprocessing immutable; override only runtime locations/device.

    `eval_datasets` is the one protocol-level override, because cross-dataset
    testing has to score a domain the checkpoint never trained on. Clip length,
    stride, crop margin, and normalization still come from the checkpoint.
    """
    config = copy.deepcopy(saved_config)
    config["device"] = runtime_config["device"]
    config["data"]["dataset_roots"] = runtime_config["data"]["dataset_roots"]
    config["train"]["report_dir"] = runtime_config["train"]["report_dir"]
    if eval_datasets:
        config["data"]["active_datasets"] = list(eval_datasets)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--mc-uncertainty-samples", type=int, default=None)
    parser.add_argument("--max-fakes-per-dataset", type=int, default=None)
    parser.add_argument("--export-embeddings", action="store_true")
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=None,
        help="Score these datasets instead of the checkpoint's training domains "
             "(use for cross-dataset testing, e.g. --eval-datasets CelebDFv3).",
    )
    parser.add_argument(
        "--recalibrate-threshold",
        action="store_true",
        help="Recalibrate the operating point on this split's real videos instead of "
             "reusing the checkpoint's. Required for meaningful cross-dataset accuracy.",
    )
    args = parser.parse_args()

    runtime_config = load_config(args.config)
    device = resolve_device(runtime_config["device"])
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_config = checkpoint["config"]
    config = checkpoint_evaluation_config(saved_config, runtime_config, args.eval_datasets)
    seed_everything(config["seed"])
    pyro.clear_param_store()
    extractor, model = build_model(saved_config, device)
    extractor.load_state_dict(checkpoint["feature_extractor"])
    pyro.get_param_store().set_state(checkpoint["pyro_params"])

    records = active_records(load_manifest(args.manifest), config)
    records = select_records(records, args.split)
    if args.max_fakes_per_dataset is not None:
        records = capped_validation_records(
            records, config["seed"], args.max_fakes_per_dataset
        )
    dataset = make_dataset(
        records,
        config,
        training=False,
        clips_per_video=config["data"]["eval_clips_per_video"],
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["physical_batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    mc_samples = config["train"].get("report_mc_uncertainty_samples", 0)
    if args.mc_uncertainty_samples is not None:
        mc_samples = args.mc_uncertainty_samples
    score_sign = float(checkpoint.get("score_sign", -1.0))
    values = score_loader(
        model, loader, device, mc_samples,
        collect_embeddings=args.export_embeddings,
        score_sign=score_sign,
    )
    # The stored threshold was calibrated on the training domain's reals, so a
    # cross-dataset run must recalibrate or its accuracy numbers are meaningless.
    # AUROC/EER are threshold-free and stay comparable either way.
    threshold = None if args.recalibrate_threshold else checkpoint["threshold"]
    if args.recalibrate_threshold:
        print("Recalibrating the threshold on this split's real videos.")
    metrics, _ = evaluate_values(
        values,
        config["train"]["calibration_false_positive_rate"],
        threshold=threshold,
        include_methods=True,
    )
    metrics.update({
        "split": args.split,
        "objective": checkpoint.get("data_protocol", {}).get("objective", "one_class_real"),
        "score_sign": score_sign,
        "trained_datasets": saved_config["data"].get("active_datasets", []),
        "evaluated_datasets": config["data"].get("active_datasets", []),
        "threshold_source": "recalibrated" if args.recalibrate_threshold else "checkpoint",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_videos": len(records),
        "num_clips_per_video": int(config["data"]["eval_clips_per_video"]),
        "clip_length": int(config["data"]["clip_length"]),
        "preprocessing_config_source": "checkpoint",
        "mean_predictive_std": float(np.mean(values["stds"])),
    })
    report_dir = ensure_dir(config["train"]["report_dir"])
    report = save_evaluation_report(values, metrics, report_dir, args.split)
    if args.export_embeddings:
        np.savez_compressed(
            report_dir / "{}_embeddings.npz".format(args.split),
            embeddings=values["embeddings"],
            labels=values["labels"],
            datasets=values["datasets"],
            paths=np.asarray(values["paths"]),
        )
    print(metrics)
    print("Report: {}".format(report.resolve()))


if __name__ == "__main__":
    main()
