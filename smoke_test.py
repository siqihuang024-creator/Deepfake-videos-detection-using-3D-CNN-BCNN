"""Check configuration, model contracts, and optionally run one expensive 3D pass."""

import argparse
import sys
from pathlib import Path

import torch
import pyro
from pyro.infer import SVI, TraceGraph_ELBO

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from video_bcnn.model import Stable3DFeatureExtractor, VideoBayesianCNN
from video_bcnn.utils import load_config, resolve_device, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--svi-step", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    extractor = Stable3DFeatureExtractor(
        config["model"]["temporal_kernel_size"], config["model"]["conv_channels"]
    ).to(device)
    result = {
        "device": str(device),
        "clip_length": int(config["data"]["clip_length"]),
        "temporal_kernel": int(extractor.temporal_kernel_size),
        "temporal_receptive_field": int(extractor.temporal_receptive_field),
        "conv_channels": list(extractor.conv_channels),
        "feature_dim": int(extractor.feature_dim),
        "extractor_parameters": sum(parameter.numel() for parameter in extractor.parameters()),
        "bayesian_fc1_weights": int(extractor.feature_dim * 512),
    }
    if args.forward or args.svi_step:
        clips = torch.zeros(
            1, 3, int(config["data"]["clip_length"]), 224, 224, device=device
        )
        if args.forward:
            extractor.eval()
            with torch.no_grad():
                features = extractor(clips)
            result["forward_shape"] = list(features.shape)
            result["forward_finite"] = bool(torch.isfinite(features).all().item())
        if args.svi_step:
            pyro.clear_param_store()
            extractor.train()
            clips = torch.randn_like(clips)
            model = VideoBayesianCNN(
                extractor,
                dropout=config["model"]["dropout"],
                prior_std=config["model"]["prior_std"],
                observation_std=config["model"]["observation_std"],
                rho_init=config["model"]["posterior_rho_init"],
                kl_weight=config["train"]["kl_weight"],
            )
            svi = SVI(
                model.model,
                model.guide,
                pyro.optim.SGD({"lr": 1e-5}),
                loss=TraceGraph_ELBO(),
            )
            before = extractor.conv1.weight.detach().clone()
            loss = float(svi.step(clips, torch.ones(1, device=device), 1))
            delta = float((extractor.conv1.weight.detach() - before).abs().max().item())
            result["svi_loss"] = loss
            result["conv1_update_max_abs"] = delta
            result["svi_updated_extractor"] = bool(delta > 0.0)
    print(result)


if __name__ == "__main__":
    main()
