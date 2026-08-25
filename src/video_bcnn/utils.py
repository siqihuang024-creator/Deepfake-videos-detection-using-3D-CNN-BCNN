"""Shared configuration, filesystem, device, and reproducibility helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


# Manifest paths are relative to these roots, so moving the project to a rented
# GPU only needs the roots repointed -- the manifests stay valid as they are.
DATASET_ROOT_ENV_VARS = {"DFD": "DFD_ROOT", "CelebDFv3": "CELEBDFV3_ROOT"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def override_dataset_roots(config, overrides=None):
    """Repoint dataset roots for the current machine.

    Precedence: `--dataset-root NAME=PATH` beats the environment variable,
    which beats the value written in the YAML.
    """
    roots = config["data"]["dataset_roots"]
    for name, variable in DATASET_ROOT_ENV_VARS.items():
        value = os.environ.get(variable)
        if value and name in roots:
            roots[name] = value
    for item in overrides or []:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(
                "--dataset-root expects NAME=PATH, received {!r}.".format(item)
            )
        if name not in roots:
            raise ValueError(
                "Unknown dataset {!r}; the config declares {}.".format(name, sorted(roots))
            )
        roots[name] = path
    return config


def override_num_workers(config, value=None):
    """Repoint the DataLoader worker count for this machine (env: NUM_WORKERS).

    The right value is machine-specific and not predictable from RAM: clip
    loading is dominated by Haar detection on full-resolution frames, which is
    memory-bandwidth bound, so extra workers can make throughput worse. Measure
    it with scripts/benchmark_loader.py on the target machine.

    The chosen value lands in the config, so every checkpoint records the
    setting the run actually used -- it perturbs augmentation RNG, and results
    are only comparable across runs that share it.
    """
    if value is None:
        variable = os.environ.get("NUM_WORKERS")
        value = int(variable) if variable else None
    if value is not None:
        if int(value) < 0:
            raise ValueError("num_workers must be >= 0, received {}.".format(value))
        config["data"]["num_workers"] = int(value)
    return config


def verify_dataset_roots(config):
    """Fail fast when an active dataset root is missing on this machine.

    A wrong root otherwise surfaces as hundreds of per-video decode errors deep
    into a training run, which is an expensive way to learn about a typo.
    """
    roots = config["data"]["dataset_roots"]
    active = config["data"].get("active_datasets") or sorted(roots)
    missing = [
        "{}: {}".format(name, roots[name])
        for name in active
        if name in roots and not Path(roots[name]).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "These dataset roots do not exist on this machine:\n  {}\n"
            "Fix data.dataset_roots in the config, export {}, or pass "
            "--dataset-root NAME=PATH.".format(
                "\n  ".join(missing),
                "/".join(DATASET_ROOT_ENV_VARS[name] for name in sorted(DATASET_ROOT_ENV_VARS)),
            )
        )
    return {name: roots[name] for name in active if name in roots}


def ensure_dir(path):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path, values):
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, allow_nan=False)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        print("CUDA is unavailable; falling back to CPU.")
    return torch.device("cpu")


def load_checkpoint(path, device):
    """Load both legacy and current PyTorch checkpoints."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)
