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


def checkpoint_evaluation_config(saved_config, runtime_config):
    """Keep trained preprocessing immutable; override only runtime locations/device."""
    config = copy.deepcopy(saved_config)
    config["device"] = runtime_config["device"]
    config["data"]["dataset_roots"] = runtime_config["data"]["dataset_roots"]
    config["train"]["report_dir"] = runtime_config["train"]["report_dir"]
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
    args = parser.parse_args()

    runtime_config = load_config(args.config)
    device = resolve_device(runtime_config["device"])
    checkpoint = load_checkpoint(args.checkpoint, device)
    saved_config = checkpoint["config"]
    config = checkpoint_evaluation_config(saved_config, runtime_config)
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
    values = score_loader(
        model, loader, device, mc_samples, collect_embeddings=args.export_embeddings
    )
    metrics, _ = evaluate_values(
        values,
        config["train"]["calibration_false_positive_rate"],
        threshold=checkpoint["threshold"],
        include_methods=True,
    )
    metrics.update({
        "split": args.split,
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
