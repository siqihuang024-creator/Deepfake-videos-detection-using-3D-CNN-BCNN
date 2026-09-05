"""Ask whether the extractor's features carry real/fake information.

This is Stage A's acceptance test, not a one-off debugging aid. Stage A hands
Stage B an extractor and throws its head away, and that head is
``fc1 -> dropout -> out`` with **no activation between the two linear layers**
(``model.py:253-255``) -- a composition of two linear maps onto one output, so
it is a single linear functional however many parameters it holds. "Did Stage A
work" therefore means exactly: can a linear classifier separate real from fake
from these features, on identities it has not seen? Training BCE cannot answer
that, and a 62-video validation split has an interval that covers chance.

Two traps this script exists to avoid, both of which its first version fell
into:

  * A probe fitted and scored on the same videos is meaningless when the
    feature dimension dwarfs the sample count. Measured: 60 videos in 15488
    dimensions gave 0.9422 from an *untrained* extractor. So the probe is
    cross-validated with folds split by identity, and ``--random-init`` keeps
    an untrained reference beside every number.

  * Comparing a per-dimension spread against a whole-vector L2 norm mixes
    scales and manufactures apparent collapse -- the norm of a d-vector grows
    like sqrt(d), which understated the spread 124-fold at d = 15488. The norm
    is divided by sqrt(feature_dim) first, so ``relative_variation`` is the
    honest "how much do videos differ, relative to how large the features are".

Usage:
    python scripts/feature_diagnostics.py \
        --checkpoint artifacts/run_stage_a_dfd_decimate_pretrain/checkpoints/best.pt \
        --manifest artifacts/manifests/combined_manifest_stage_a.csv \
        --split train --max-videos 60 --random-init
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from video_bcnn.data import load_manifest  # noqa: E402
from video_bcnn.experiment import (  # noqa: E402
    active_records, make_dataset, select_records,
)
from video_bcnn.utils import (  # noqa: E402
    override_dataset_roots, resolve_device,
)
from train_3d_bcnn import build_model  # noqa: E402

# Ridge strengths as multiples of the Gram matrix's mean diagonal, so one grid
# means the same thing whatever the feature dimension turns out to be.
RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def auroc(labels, scores):
    """Mann-Whitney U, so the script does not depend on scikit-learn."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    positive, negative = scores[labels == 1], scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average the ranks inside each tie group, otherwise duplicated scores --
    # which is what near-constant features produce -- bias the statistic.
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rank_sum = ranks[:len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2.0)
                 / (len(positive) * len(negative)))


def collect_features(extractor, dataset, device):
    """One mean-pooled feature vector per video, with its label and identity."""
    features, labels, identities = [], [], []
    extractor.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            if item is None:
                continue
            vector = extractor(item["clips"].to(device)).double().mean(dim=0).cpu()
            features.append(vector)
            labels.append(int(item["label"]))
            # Falling back to the path keeps every video in its own fold rather
            # than silently merging them when a manifest carries no identity.
            identities.append(item.get("target_id") or item["path"])
            if (index + 1) % 10 == 0 or index + 1 == len(dataset):
                print("  {}/{} videos".format(index + 1, len(dataset)), flush=True)
    if not features:
        raise RuntimeError("No video was readable.")
    return torch.stack(features), np.asarray(labels), identities


def ridge_scores(train_x, train_y, test_x, strength):
    """Closed-form ridge in the dual, so an n x n system is solved, not d x d.

    With 48 training videos and 15488 features the primal is hopeless and any
    iterative fit brings a learning rate to argue about. The dual is exact:
    w = X'a with a = (XX' + lambda I)^-1 y, and predictions never form w.
    Standardisation uses the training fold's statistics alone, or the held-out
    videos would leak into their own normalisation.
    """
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0).clamp_min(1e-8)
    a = (train_x - mean) / scale
    b = (test_x - mean) / scale
    centred = train_y - train_y.mean()
    gram = a @ a.T
    lam = strength * float(torch.diagonal(gram).mean())
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    alpha = torch.linalg.solve(gram + lam * eye, centred)
    return (b @ a.T) @ alpha


def probe(features, labels, identities, folds=5, seed=42):
    """Out-of-fold AUROC of a linear probe, with folds split by identity.

    Held out by identity rather than by video because clips of one actor are
    not independent, and the question is whether the features generalise to a
    face the extractor has never seen.
    """
    x = features.double()
    y = torch.tensor(labels.astype("float64"))
    unique = sorted(set(identities))
    generator = np.random.RandomState(seed)
    order = generator.permutation(len(unique))
    fold_of_identity = {unique[position]: index % folds
                        for index, position in enumerate(order)}
    fold = np.asarray([fold_of_identity[name] for name in identities])
    usable = min(folds, len(unique))

    results = {}
    for strength in RIDGE_GRID:
        out_of_fold = np.full(len(labels), np.nan)
        for index in range(usable):
            test = fold == index
            train = ~test
            if test.sum() == 0 or len(np.unique(labels[train])) < 2:
                continue
            out_of_fold[test] = ridge_scores(
                x[train], y[train], x[test], strength).numpy()
        scored = ~np.isnan(out_of_fold)
        if scored.sum() < 4 or len(np.unique(labels[scored])) < 2:
            continue
        results[strength] = {
            "out_of_fold": auroc(labels[scored], out_of_fold[scored]),
            "in_sample": auroc(labels, ridge_scores(x, y, x, strength).numpy()),
            "scored": int(scored.sum()),
        }
    return results, len(unique)


def describe(features, labels, name):
    dimension = features.shape[1]
    norms = features.norm(dim=1)
    # A per-dimension quantity needs a per-dimension reference: the L2 norm of
    # a d-vector grows like sqrt(d), so dividing a per-dimension spread by the
    # raw norm understates it 124-fold at d = 15488.
    per_dimension_rms = float(norms.mean()) / math.sqrt(dimension)
    spread = float(features.std(dim=0).mean())
    report = {
        "videos": int(len(features)),
        "real": int((labels == 1).sum()),
        "fake": int((labels == 0).sum()),
        "mean_feature_norm": float(norms.mean()),
        "per_dimension_rms": per_dimension_rms,
        "between_video_std": spread,
        "relative_variation": spread / max(per_dimension_rms, 1e-12),
    }
    real, fake = features[labels == 1], features[labels == 0]
    if len(real) and len(fake):
        centroid_real, centroid_fake = real.mean(dim=0), fake.mean(dim=0)
        gap = float((centroid_real - centroid_fake).norm())
        within = float(torch.cat([
            (real - centroid_real).norm(dim=1),
            (fake - centroid_fake).norm(dim=1)]).mean())
        report["centroid_l2"] = gap
        report["within_class_radius"] = within
        # Scale-free: the class gap in units of within-class scatter, so a
        # uniform growth in feature magnitude leaves it unchanged.
        report["separation_ratio"] = gap / max(within, 1e-12)
        cosine = torch.dot(centroid_real, centroid_fake) / (
            centroid_real.norm() * centroid_fake.norm() + 1e-12)
        report["centroid_cosine_distance"] = float(1.0 - cosine)
    print("\n--- {} ---".format(name))
    for key, value in report.items():
        print("  {:<26} {}".format(
            key, value if isinstance(value, int) else "{:.6g}".format(value)))
    return report


def report_probe(features, labels, identities, name):
    results, identity_count = probe(features, labels, identities)
    print("\n  linear probe on {} identities, folds split by identity".format(
        identity_count))
    print("  {:>10}  {:>15}  {:>12}".format("ridge", "held-out AUROC", "in-sample"))
    best = None
    for strength in sorted(results):
        entry = results[strength]
        print("  {:>10.0e}  {:>15.4f}  {:>12.4f}".format(
            strength, entry["out_of_fold"], entry["in_sample"]))
        if best is None or entry["out_of_fold"] > best[1]:
            best = (strength, entry["out_of_fold"])
    if best is None:
        print("  too few identities or one class only; no probe fitted [{}]".format(name))
        return float("nan")
    print("  best held-out AUROC {:.4f} at ridge {:.0e}   [{}]".format(
        best[1], best[0], name))
    return best[1]


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
    parser.add_argument("--save-features", default=None,
                        help="Write the extracted features to this .npz so the "
                             "probe can be re-analysed without decoding again.")
    parser.add_argument(
        "--random-init", action="store_true",
        help="Also report an untrained extractor. Without that reference a high "
             "probe score cannot be told apart from the free separability any "
             "high-dimensional feature offers.")
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
    features, labels, identities = collect_features(extractor, dataset, device)
    describe(features, labels, "trained extractor")
    trained = report_probe(features, labels, identities, "trained")

    if args.random_init:
        fresh, _ = build_model(config, device)
        print("\nextracting features with an untrained extractor...")
        base_features, base_labels, base_identities = collect_features(
            fresh, dataset, device)
        describe(base_features, base_labels, "random init (reference)")
        baseline = report_probe(base_features, base_labels, base_identities,
                                "random init")
        print("\ntraining moved the held-out probe by {:+.4f}  ({:.4f} -> {:.4f})"
              .format(trained - baseline, baseline, trained))

    if args.save_features:
        path = Path(args.save_features)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=features.numpy(), labels=labels,
                            identities=np.asarray(identities))
        print("\nwrote {}".format(path))

    print("\n" + "=" * 70)
    print("How to read this:")
    print("  held-out AUROC is the number that matters. In-sample is printed")
    print("  only to show how much free separability the dimension supplies.")
    print("  trained ~ random init  -> the convolutions learned nothing usable")
    print("  trained >> random init -> the features do carry class information")
    print("                            and the training loop is what is failing")
    print("  relative_variation ~ 0 -> collapse: every video maps to one point")
    print("  separation_ratio       -> class gap in units of within-class scatter")
    return 0


if __name__ == "__main__":
    sys.exit(main())
