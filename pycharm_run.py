"""PyCharm entry point: select TASK and press Run."""

import subprocess
import sys
from pathlib import Path

import yaml


# Options:
#   build_manifest, build_supervised_manifest,
#   smoke_test, smoke_test_forward, smoke_test_svi,
#   train_<experiment>,
#   evaluate_<experiment>_val,       evaluate_<experiment>_test,
#   evaluate_<experiment>_cross_val, evaluate_<experiment>_cross_test
#
# "cross" scores the datasets the checkpoint was NOT trained on and recalibrates
# the operating point there, which is the generalization number for the
# supervised experiments.
TASK = "train_dfd_supervised_3d"

# CelebDFv3 has 10433 test fakes; a full cross-dataset pass is expensive.
# Set an integer here for a method-stratified debug subset, None for the
# complete split that final results must use.
DEBUG_MAX_FAKES_PER_DATASET = None

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACTS = PROJECT_ROOT / "artifacts"
CONFIG_3D = CONFIG_DIR / "combined_3d_bcnn.yaml"
ONE_CLASS_MANIFEST = ARTIFACTS / "manifests" / "combined_manifest_dfd_donor_safe.csv"
SUPERVISED_MANIFEST = ARTIFACTS / "manifests" / "combined_manifest_supervised.csv"

ALL_DATASETS = ("DFD", "CelebDFv3")


def experiment(config_name, run_name, manifest, datasets):
    return {
        "config": CONFIG_DIR / config_name,
        "checkpoint": ARTIFACTS / run_name / "checkpoints" / "best.pt",
        "manifest": manifest,
        "datasets": tuple(datasets),
    }


EXPERIMENTS = {
    # Real-only one-class protocol.
    "dfd_2d": experiment(
        "dfd_2d_control.yaml", "run_dfd_2d_bcnn_matched", ONE_CLASS_MANIFEST, ["DFD"]),
    "dfd_3d": experiment(
        "dfd_3d_bcnn.yaml", "run_dfd_3d_bcnn_t8_k3", ONE_CLASS_MANIFEST, ["DFD"]),
    "combined_2d": experiment(
        "combined_2d_control.yaml", "run_2d_bcnn_matched_combined", ONE_CLASS_MANIFEST, ALL_DATASETS),
    "combined_3d": experiment(
        "combined_3d_bcnn.yaml", "run_3d_bcnn_t8_k3_combined", ONE_CLASS_MANIFEST, ALL_DATASETS),
    "combined_3d_t16_k5_wide": experiment(
        "combined_3d_bcnn_t16_k5_wide.yaml", "run_3d_bcnn_t16_k5_wide_combined",
        ONE_CLASS_MANIFEST, ALL_DATASETS),
    # Supervised protocol: train per dataset, then self- and cross-test.
    "dfd_supervised_3d": experiment(
        "dfd_supervised_3d.yaml", "run_dfd_supervised_3d_t8_k3", SUPERVISED_MANIFEST, ["DFD"]),
    "celeb_supervised_3d": experiment(
        "celeb_supervised_3d.yaml", "run_celeb_supervised_3d_t8_k3",
        SUPERVISED_MANIFEST, ["CelebDFv3"]),
}


def run(command):
    printable = " ".join('"{}"'.format(item) if " " in str(item) else str(item) for item in command)
    print("Running:", printable)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def build_manifest(config):
    run([
        sys.executable,
        "scripts/build_manifest.py",
        "--dfd-root", config["data"]["dataset_roots"]["DFD"],
        "--celeb-root", config["data"]["dataset_roots"]["CelebDFv3"],
        "--output-dir", "artifacts/manifests",
        "--seed", str(config["seed"]),
    ])


def build_supervised_manifest():
    run([
        sys.executable,
        "scripts/build_supervised_manifest.py",
        "--source", str(ONE_CLASS_MANIFEST),
        "--output", str(SUPERVISED_MANIFEST),
    ])


def lookup(name):
    if name not in EXPERIMENTS:
        raise ValueError("Unknown experiment {!r}; expected one of {}.".format(
            name, sorted(EXPERIMENTS)
        ))
    return EXPERIMENTS[name]


def require_manifest(selected, config):
    manifest = selected["manifest"]
    if manifest.is_file():
        return manifest
    if not ONE_CLASS_MANIFEST.is_file():
        print("Strict combined manifest is missing; building it first.")
        build_manifest(config)
    if manifest == SUPERVISED_MANIFEST:
        print("Supervised manifest is missing; deriving it from the donor-safe manifest.")
        build_supervised_manifest()
    return manifest


def evaluate(name, split, cross, config):
    selected = lookup(name)
    checkpoint = selected["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError("No matching best checkpoint; train this experiment first.")
    manifest = require_manifest(selected, config)
    command = [
        sys.executable, "evaluate_3d_bcnn.py",
        "--config", str(selected["config"]), "--manifest", str(manifest),
        "--checkpoint", str(checkpoint), "--split", split,
    ]
    if cross:
        held_out = [item for item in ALL_DATASETS if item not in selected["datasets"]]
        if not held_out:
            raise ValueError(
                "{!r} already trains on every dataset, so it has no cross-dataset split.".format(name)
            )
        command += ["--eval-datasets"] + held_out + ["--recalibrate-threshold"]
    if DEBUG_MAX_FAKES_PER_DATASET is not None:
        command += ["--max-fakes-per-dataset", str(DEBUG_MAX_FAKES_PER_DATASET)]
    run(command)


def main():
    with open(CONFIG_3D, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if TASK == "build_manifest":
        build_manifest(config)
        return
    if TASK == "build_supervised_manifest":
        build_supervised_manifest()
        return
    if TASK in ("smoke_test", "smoke_test_forward", "smoke_test_svi"):
        command = [sys.executable, "smoke_test.py", "--config", str(CONFIG_3D)]
        if TASK == "smoke_test_forward":
            command.append("--forward")
        if TASK == "smoke_test_svi":
            command.append("--svi-step")
        run(command)
        return
    if TASK.startswith("train_"):
        selected = lookup(TASK[len("train_"):])
        manifest = require_manifest(selected, config)
        run([
            sys.executable, "train_3d_bcnn.py",
            "--config", str(selected["config"]), "--manifest", str(manifest),
        ])
        return
    if TASK.startswith("evaluate_"):
        name = TASK[len("evaluate_"):]
        for suffix, split, cross in (
            ("_cross_val", "val", True),
            ("_cross_test", "test", True),
            ("_val", "val", False),
            ("_test", "test", False),
        ):
            if name.endswith(suffix):
                evaluate(name[:-len(suffix)], split, cross, config)
                return
        raise ValueError("Evaluation TASK must end in _val, _test, _cross_val, or _cross_test.")
    raise ValueError("Unknown TASK: {}".format(TASK))


if __name__ == "__main__":
    main()
