"""PyCharm entry point: select TASK and press Run."""

import subprocess
import sys
from pathlib import Path

import yaml


# Options: build_manifest, smoke_test, smoke_test_forward, smoke_test_svi,
# train_dfd_2d, train_dfd_3d, train_combined_2d, train_combined_3d,
# train_combined_3d_t16_k5_wide,
# evaluate_<experiment>_val, evaluate_<experiment>_test.
TASK = "train_combined_3d_t16_k5_wide"

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_3D = PROJECT_ROOT / "configs" / "combined_3d_bcnn.yaml"
MANIFEST = PROJECT_ROOT / "artifacts" / "manifests" / "combined_manifest_dfd_donor_safe.csv"
EXPERIMENTS = {
    "dfd_2d": (
        PROJECT_ROOT / "configs" / "dfd_2d_control.yaml",
        PROJECT_ROOT / "artifacts" / "run_dfd_2d_bcnn_matched" / "checkpoints" / "best.pt",
    ),
    "dfd_3d": (
        PROJECT_ROOT / "configs" / "dfd_3d_bcnn.yaml",
        PROJECT_ROOT / "artifacts" / "run_dfd_3d_bcnn_t8_k3" / "checkpoints" / "best.pt",
    ),
    "combined_2d": (
        PROJECT_ROOT / "configs" / "combined_2d_control.yaml",
        PROJECT_ROOT / "artifacts" / "run_2d_bcnn_matched_combined" / "checkpoints" / "best.pt",
    ),
    "combined_3d": (
        CONFIG_3D,
        PROJECT_ROOT / "artifacts" / "run_3d_bcnn_t8_k3_combined" / "checkpoints" / "best.pt",
    ),
    "combined_3d_t16_k5_wide": (
        PROJECT_ROOT / "configs" / "combined_3d_bcnn_t16_k5_wide.yaml",
        PROJECT_ROOT / "artifacts" / "run_3d_bcnn_t16_k5_wide_combined" / "checkpoints" / "best.pt",
    ),
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


def main():
    with open(CONFIG_3D, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if TASK == "build_manifest":
        build_manifest(config)
        return
    if TASK in ("smoke_test", "smoke_test_forward", "smoke_test_svi"):
        command = [sys.executable, "smoke_test.py", "--config", str(CONFIG_3D)]
        if TASK == "smoke_test_forward":
            command.append("--forward")
        if TASK == "smoke_test_svi":
            command.append("--svi-step")
        run(command)
        return
    if not MANIFEST.is_file():
        print("Strict combined manifest is missing; building it first.")
        build_manifest(config)
    if TASK.startswith("train_"):
        experiment = TASK.replace("train_", "")
        if experiment not in EXPERIMENTS:
            raise ValueError("Unknown training experiment: {}".format(experiment))
        selected_config, _ = EXPERIMENTS[experiment]
        run([sys.executable, "train_3d_bcnn.py", "--config", str(selected_config), "--manifest", str(MANIFEST)])
        return
    if TASK.startswith("evaluate_"):
        if not TASK.endswith(("_val", "_test")):
            raise ValueError("Evaluation TASK must end in _val or _test.")
        split = "val" if TASK.endswith("_val") else "test"
        suffix = "_{}".format(split)
        experiment = TASK[len("evaluate_"):-len(suffix)]
        if experiment not in EXPERIMENTS:
            raise ValueError("Unknown evaluation experiment: {}".format(experiment))
        selected_config, checkpoint = EXPERIMENTS[experiment]
        if not checkpoint.is_file():
            raise FileNotFoundError("No matching best checkpoint; train this experiment first.")
        run([
            sys.executable, "evaluate_3d_bcnn.py",
            "--config", str(selected_config), "--manifest", str(MANIFEST),
            "--checkpoint", str(checkpoint), "--split", split,
        ])
        return
    raise ValueError("Unknown TASK: {}".format(TASK))


if __name__ == "__main__":
    main()
