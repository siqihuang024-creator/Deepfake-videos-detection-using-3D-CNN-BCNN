"""Ask whether the extractor's output separates real from fake at all.

Stage A trains an extractor behind ``DeterministicHead``, and that head is
``fc1 -> dropout -> out`` with **no activation between the two linear layers**
(``model.py:253-255``). A composition of two linear maps onto one output is a
single linear functional, so the head's 7.9M parameters are an
over-parameterised way of writing one 15488-dimensional weight vector. Stage A
is therefore already a linear probe on the extractor's features, and a training
BCE pinned at ln 2 says one thing precisely: **those features are not linearly
separable**, even while the extractor is being trained to make them so.

That is surprising enough to check directly. Any n points in general position
in R^15488 are linearly separable when n is far below 15488, and the 4-identity
diagnostic had 71 videos. A model that still cannot fit them is more likely to
be emitting nearly the same vector for every video -- collapse -- than to be
facing a hard problem.

So this script loads a checkpoint, extracts one feature vector per video, and
reports three things:

  1. Collapse. Within-class variance against the mean feature norm. A ratio
     near zero means every video maps to the same point.
  2. Separation. Distance between the real and fake centroids, in units of the
     within-class spread -- a Fisher-style ratio, unlike the raw L2 in
     ``_feature_diagnostics`` which says nothing without a scale to read it in.
  3. Linear separability. A logistic regression fitted on these very features
     and scored on them. This is the ceiling Stage A's linear head could ever
     reach. If the probe reaches 1.0 while training sat at ln 2, the features
     are fine and the failure is in optimising through the convolutions; if the
     probe also fails, the representation is the problem.

``--random-init`` repeats everything with a freshly initialised extractor. That
is the reference that matters: if training moved none of these numbers, the
convolutions learned nothing at all.

Usage:
    python scripts/feature_diagnostics.py \
        --checkpoint artifacts/run_stage_a_dfd_decimate_pretrain/checkpoints/best.pt \
        --manifest artifacts/manifests/combined_manifest_stage_a.csv \
        --split val --max-videos 60 --random-init
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from video_bcnn.experiment import (  # noqa: E402
    active_records, make_dataset, select_records,
)
from video_bcnn.data import load_manifest  # noqa: E402
from video_bcnn.utils import (  # noqa: E402
    override_dataset_roots, resolve_device,
)
from train_3d_bcnn import build_model  # noqa: E402


def auroc(labels, scores):
    """Mann-Whitney U, so the script does not depend on scikit-learn."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average the ranks inside each tie group, otherwise duplicated scores -- which
    # is exactly what collapsed features produce -- bias the statistic.
    values = np.concatenate([positive, negative])
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[:len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2.0)
                 / (len(positive) * len(negative)))


def collect_features(extractor, dataset, device, limit=None):
    """One mean-pooled feature vector per video, plus its label."""
    features, labels, paths = [], [], []
    extractor.eval()
    total = len(dataset) if limit is None else min(limit, len(dataset))
    with torch.no_grad():
        for index in range(total):
            item = dataset[index]
            if item is None:
                continue
            clips = item["clips"].to(device)
            vector = extractor(clips).double().mean(dim=0).cpu()
            features.append(vector)
            labels.append(int(item["label"]))
            paths.append(item["path"])
            if (index + 1) % 10 == 0 or index + 1 == total:
                print("  {}/{} videos".format(index + 1, total), flush=True)
    if not features:
        raise RuntimeError("No video was readable.")
    return torch.stack(features), np.asarray(labels), paths


def linear_probe(features, labels, steps=400, weight_decay=1e-3, seed=42):
    """Best AUROC a linear map on these features can reach, fitted on them.

    Fitted and scored on the same videos on purpose: the question is whether a
    linear separator exists at all, not whether it generalises. Stage A's head
    is linear, so this is the ceiling that head could ever have hit.
    """
    torch.manual_seed(seed)
    x = features.float()
    x = (x - x.mean(dim=0)) / x.std(dim=0).clamp_min(1e-6)
    y = torch.tensor((labels == 1).astype("float32"))
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.Adam([weight, bias], lr=0.05)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        logits = x @ weight + bias
        loss = loss_fn(logits, y) + weight_decay * weight.square().sum()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        scores = (x @ weight + bias).numpy()
    return auroc(labels, scores), float(loss.detach())


def describe(features, labels, name):
    real = features[labels == 1]
    fake = features[labels == 0]
    report = {"videos": int(len(features)), "real": int(len(real)),
              "fake": int(len(fake))}
    norms = features.norm(dim=1)
    report["mean_feature_norm"] = float(norms.mean())
    # Spread between videos, relative to how large the vectors are. A collapsed
    # extractor emits one point, so this ratio goes to zero however large the
    # raw variance looks in absolute terms.
    spread = features.std(dim=0).mean()
    report["between_video_std"] = float(spread)
    report["spread_over_norm"] = float(spread / max(norms.mean(), 1e-12))
    if len(real) and len(fake):
        centroid_real, centroid_fake = real.mean(dim=0), fake.mean(dim=0)
        gap = float((centroid_real - centroid_fake).norm())
        within = float(torch.cat([
            (real - centroid_real).norm(dim=1),
            (fake - centroid_fake).norm(dim=1)]).mean())
        report["centroid_l2"] = gap
        report["within_class_radius"] = within
        # Fisher-style: how far apart the classes sit measured in units of how
        # scattered each class is. Below ~0.2 the classes overlap almost fully.
        report["separation_ratio"] = gap / max(within, 1e-12)
        cosine = torch.dot(centroid_real, centroid_fake) / (
            centroid_real.norm() * centroid_fake.norm() + 1e-12)
        report["centroid_cosine_distance"] = float(1.0 - cosine)
    print("\n--- {} ---".format(name))
    for key, value in report.items():
        print("  {:<26} {}".format(
            key, value if isinstance(value, int) else "{:.6g}".format(value)))
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-videos", type=int, default=60)
    parser.add_argument("--clips-per-video", type=int, default=2)
    parser.add_argument("--dataset-root", action="append", default=None,
                        metavar="NAME=PATH")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--random-init", action="store_true",
        help="Also report an untrained extractor, so the trained numbers have "
             "a reference. If they match, the convolutions learned nothing.")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = override_dataset_roots(payload["config"], args.dataset_root)
    device = resolve_device(args.device or config.get("device", "cpu"))
    print("checkpoint epoch {}, recorded validation AUROC {:.4f}".format(
        payload.get("epoch"), payload.get("validation_auroc", float("nan"))))
    print("architecture: activation={} norm={} conv_channels={} "
          "spatial_output_size={} feature_dim={}".format(
              config["model"].get("activation", "sigmoid"),
              config["model"].get("norm", "none"),
              config["model"]["conv_channels"],
              config["model"].get("spatial_output_size", 22),
              config["model"].get("feature_dim")))

    records = select_records(active_records(load_manifest(args.manifest), config),
                             args.split)
    reals = [row for row in records if int(row["label"]) == 1]
    fakes = [row for row in records if int(row["label"]) == 0]
    half = max(args.max_videos // 2, 1)
    records = reals[:half] + fakes[:half]
    print("scoring {} videos from split {!r} ({} real, {} fake), {} clips each"
          .format(len(records), args.split, len(reals[:half]), len(fakes[:half]),
                  args.clips_per_video))

    config["data"]["num_workers"] = 0
    dataset = make_dataset(records, config, training=False,
                           clips_per_video=args.clips_per_video)

    extractor, _ = build_model(config, device)
    extractor.load_state_dict(payload["extractor"])
    print("\nextracting features with the trained extractor...")
    features, labels, _ = collect_features(extractor, dataset, device)
    describe(features, labels, "trained extractor")
    probe, probe_loss = linear_probe(features, labels)
    print("  {:<26} {:.4f}   (BCE {:.4f})".format(
        "linear probe AUROC", probe, probe_loss))

    if args.random_init:
        fresh, _ = build_model(config, device)
        print("\nextracting features with an untrained extractor...")
        base_features, base_labels, _ = collect_features(fresh, dataset, device)
        describe(base_features, base_labels, "random init (reference)")
        base_probe, base_loss = linear_probe(base_features, base_labels)
        print("  {:<26} {:.4f}   (BCE {:.4f})".format(
            "linear probe AUROC", base_probe, base_loss))
        print("\ntraining moved the linear probe by {:+.4f}".format(probe - base_probe))

    print("\n" + "=" * 68)
    print("How to read this:")
    print("  spread_over_norm near 0      -> collapse: every video maps to one point")
    print("  separation_ratio below ~0.2  -> the classes overlap almost entirely")
    print("  linear probe AUROC near 1.0  -> features ARE separable; the failure")
    print("                                  is in optimising through the convs")
    print("  linear probe AUROC near 0.5  -> the representation itself carries")
    print("                                  no class information")
    return 0


if __name__ == "__main__":
    sys.exit(main())
