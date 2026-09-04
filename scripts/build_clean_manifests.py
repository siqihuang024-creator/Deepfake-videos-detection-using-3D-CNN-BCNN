#!/usr/bin/env python
"""Rebuild the supervised manifest with two evaluation defects removed.

Defect 1 - donor leakage on the evaluation side. The supervised manifest keeps
a fake out of *training* unless its target and donor identities are both
training identities, but it never applied the mirror rule to val/test: 41% of
CelebDFv3 validation fakes and 46% of its test fakes wear a face belonging to a
training identity. DFD already had both sides filtered (its excluded_donor_train
bucket holds exactly those evaluation rows), so this only moves CelebDFv3 rows,
into the same bucket name DFD already uses.

Defect 2 - aspect ratio predicts the label. CelebDFv3 reals are 93.9% wide and
its fakes are 65.1% square, so on any letterbox canvas the padding share alone
separates the classes: 0.83 AUROC on the validation split the runs actually
scored, against 0.819 for the trained network. Restricting to wide videos is not
enough - the residual is still 0.57 - because the wide cluster spans 1.70 to
2.15 and the two classes are not distributed alike inside it. Matching each real
to fakes of the *identical* resolution drives the padding-only AUROC to exactly
0.5, which is what the second output is for.

Training rows are copied through untouched in both outputs, so extractors
already pre-trained against the original manifest stay valid.
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTRA_COLUMNS = ["width", "height", "aspect_ratio", "shape_group", "pad_share"]
DONOR_BUCKET = "excluded_donor_train"
SHAPE_BUCKET = "excluded_shape_unmatched"


def stable_rng(seed, *parts):
    """Mirror experiment._stable_rng so subset choices survive a rerun."""
    key = ":".join([str(seed)] + [str(part) for part in parts]).encode("utf-8")
    return np.random.RandomState(int(hashlib.sha256(key).hexdigest()[:8], 16))


def normalise(path):
    return path.replace("\\", "/").lstrip("/")


def load_sizes(path):
    """Map (dataset, manifest-relative path) -> (width, height).

    video_sizes.csv writes the corpus name as the on-disk folder (DFD-Kaggle)
    and prefixes CelebDFv3 rows with a redundant directory level; the manifest
    does neither. The header also carries a BOM, hence utf-8-sig.
    """
    sizes = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "ok":
                continue
            key = normalise(row["relative_path"])
            if row["dataset"] == "CelebDFv3":
                sizes[("CelebDFv3", key[len("CelebDFv3/"):])] = (
                    int(row["width"]), int(row["height"]))
            else:
                sizes[("DFD", key)] = (int(row["width"]), int(row["height"]))
    return sizes


def pad_share(width, height, canvas):
    """Fraction of the canvas left as replicated border after a fit-inside scale."""
    scale = min(canvas[0] / width, canvas[1] / height)
    return 1.0 - (width * scale) * (height * scale) / (canvas[0] * canvas[1])


def annotate(rows, sizes, canvas, min_wide_aspect):
    """Attach resolution columns; abort rather than guess at a missing video."""
    missing = [row for row in rows if (row["dataset"], normalise(row["path"])) not in sizes]
    if missing:
        raise SystemExit(
            "{} manifest rows have no entry in the size report, first: {}".format(
                len(missing), missing[0]["path"]))
    for row in rows:
        width, height = sizes[(row["dataset"], normalise(row["path"]))]
        aspect = width / height
        row["width"] = str(width)
        row["height"] = str(height)
        row["aspect_ratio"] = "{:.4f}".format(aspect)
        # The corpus is bimodal with an empty band between 1.6 and 1.7, so any
        # cut inside it produces the same two groups.
        row["shape_group"] = "wide" if aspect >= min_wide_aspect else "square"
        row["pad_share"] = "{:.4f}".format(pad_share(width, height, canvas))
    return rows


def identity_home(rows):
    """Which split an identity's real videos live in."""
    return {(row["dataset"], row["target_id"]): row["split"]
            for row in rows if row["class_name"] == "real"}


def drop_leaked_donors(rows):
    """Move evaluation fakes whose donor face was seen in training."""
    home = identity_home(rows)
    moved = collections.Counter()
    for row in rows:
        if row["split"] not in ("val", "test") or int(row["label"]) != 0:
            continue
        if not row["donor_id"]:
            continue
        if home.get((row["dataset"], row["donor_id"])) == "train":
            moved[(row["dataset"], row["split"])] += 1
            row["split"] = DONOR_BUCKET
    return moved


def auroc_standard_error(n_real, n_fake, reference=0.8):
    """Hanley-McNeil standard error, used only to compare candidate subsets.

    Raising the fake:real ratio sharpens one side but costs whole resolutions
    that cannot supply the ratio, and reals are the scarce class, so the best
    trade-off differs per split and cannot be fixed by a constant. Evaluating
    the closed form at one reference AUROC ranks the candidates without needing
    a model to have been scored yet; the ranking is insensitive to the
    reference over any plausible range.
    """
    if n_real < 2 or n_fake < 2:
        return float("inf")
    q1 = reference / (2.0 - reference)
    q2 = 2.0 * reference ** 2 / (1.0 + reference)
    numerator = (reference * (1.0 - reference)
                 + (n_fake - 1) * (q1 - reference ** 2)
                 + (n_real - 1) * (q2 - reference ** 2))
    return (numerator / (n_fake * n_real)) ** 0.5


def match_on_resolution(rows, max_ratio, seed):
    """Keep a wide subset whose padding distribution is identical across classes.

    Within one resolution every video pads by the same amount, so a padding-only
    ranker can do no better than tie. Sampling the same fake:real ratio in every
    resolution keeps that true across strata as well, which is what pins the
    padding-only AUROC at 0.5 rather than merely reducing it.

    The ratio is chosen per split rather than imposed. DFD is a single
    resolution and needs no control at all, but it holds fewer fakes than reals,
    so a fixed 4:1 would have discarded the corpus entire.
    """
    kept, report = set(), {}
    for dataset in sorted({row["dataset"] for row in rows}):
        for split in ("val", "test"):
            # Nothing to control where padding already carries no label
            # information; subsetting such a split would only cost power. DFD is
            # 1920x1080 throughout and lands here.
            shortcut, _, _ = padding_auroc(rows, dataset, split)
            if abs(shortcut - 0.5) < 1e-9:
                for row in rows:
                    if row["dataset"] == dataset and row["split"] == split:
                        kept.add(id(row))
                report["{}:{}".format(dataset, split)] = {"passed_through": True}
                continue

            strata = collections.defaultdict(lambda: {"real": [], "fake": []})
            for row in rows:
                if (row["dataset"] != dataset or row["split"] != split
                        or row["shape_group"] != "wide"):
                    continue
                side = "real" if int(row["label"]) == 1 else "fake"
                strata[(row["width"], row["height"])][side].append(row)
            usable = {key: group for key, group in strata.items()
                      if group["real"] and group["fake"]}

            def totals(ratio):
                reals = fakes = 0
                for group in usable.values():
                    if len(group["fake"]) >= ratio * len(group["real"]):
                        reals += len(group["real"])
                        fakes += ratio * len(group["real"])
                return reals, fakes

            candidates = [(auroc_standard_error(*totals(k)), -k, k)
                          for k in range(1, max_ratio + 1)]
            error, _, ratio = min(candidates)
            reals = fakes = 0
            for key in sorted(usable):
                group = usable[key]
                want = ratio * len(group["real"])
                if len(group["fake"]) < want:
                    continue
                order = stable_rng(seed, dataset, split, key).permutation(len(group["fake"]))
                for row in group["real"]:
                    kept.add(id(row))
                for index in order[:want]:
                    kept.add(id(group["fake"][int(index)]))
                reals += len(group["real"])
                fakes += want
            report["{}:{}".format(dataset, split)] = {
                "wide_candidates": sum(len(v["real"]) + len(v["fake"])
                                       for v in strata.values()),
                "resolutions_with_both_classes": len(usable),
                "chosen_ratio": ratio,
                "matched_real": reals,
                "matched_fake": fakes,
                "expected_auroc_standard_error": round(error, 4),
            }
    dropped = collections.Counter()
    for row in rows:
        if row["split"] in ("val", "test") and id(row) not in kept:
            dropped[(row["dataset"], row["split"])] += 1
            row["split"] = SHAPE_BUCKET
    return report, dropped


def padding_auroc(rows, dataset, split):
    """AUROC of a ranker that reads nothing but the padding share."""
    pads = {0: [], 1: []}
    for row in rows:
        if row["dataset"] == dataset and row["split"] == split:
            pads[int(row["label"])].append(float(row["pad_share"]))
    fake, real = np.array(pads[0]), np.array(pads[1])
    if not len(fake) or not len(real):
        return float("nan"), len(real), len(fake)
    wins = (fake[:, None] > real[None, :]).sum()
    ties = (fake[:, None] == real[None, :]).sum()
    return (wins + 0.5 * ties) / (len(fake) * len(real)), len(real), len(fake)


def audit(rows, label):
    """Report the shortcut, the shape mix behind it, and method coverage.

    The wide share of each class is printed alongside the AUROC because the two
    repairs interact: removing fakes whose donor was seen in training removes
    face-swap and reenactment rows, which are the wide ones, so the identity fix
    on its own leaves the fake side more square than it found it.
    """
    print("\n{}".format(label))
    print("  {:<10} {:<5} {:>5} {:>6} {:>10} {:>10} {:>7} {:>19}".format(
        "dataset", "split", "real", "fake", "wide real", "wide fake",
        "methods", "padding-only AUROC"))
    for dataset in sorted({row["dataset"] for row in rows}):
        for split in ("val", "test"):
            value, reals, fakes = padding_auroc(rows, dataset, split)
            chosen = [row for row in rows
                      if row["dataset"] == dataset and row["split"] == split]
            wide = collections.Counter(
                row["class_name"] for row in chosen if row["shape_group"] == "wide")
            methods = {row["method"] for row in chosen if int(row["label"]) == 0}
            share = lambda part, whole: "n/a" if not whole else "{:.0%}".format(part / whole)
            print("  {:<10} {:<5} {:>5} {:>6} {:>10} {:>10} {:>7} {:>19}".format(
                dataset, split, reals, fakes,
                share(wide["real"], reals), share(wide["fake"], fakes), len(methods),
                "n/a" if np.isnan(value) else "{:.4f}".format(value)))


def write(rows, path, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("  wrote {} ({} rows)".format(path, len(rows)))


def method_shapes(rows):
    """Wide/square counts per CelebDFv3 forgery method across val and test."""
    shapes = collections.defaultdict(lambda: {"wide": 0, "square": 0})
    for row in rows:
        if (row["dataset"] == "CelebDFv3" and int(row["label"]) == 0
                and row["split"] in ("val", "test")):
            shapes[row["method"]][row["shape_group"]] += 1
    return dict(shapes)


def counts(rows):
    tally = collections.Counter(
        (row["dataset"], row["split"], row["class_name"]) for row in rows)
    return {"{}:{}:{}".format(*key): value for key, value in sorted(tally.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest",
                        default=str(ROOT / "artifacts/manifests/combined_manifest_supervised.csv"))
    parser.add_argument("--sizes",
                        default=str(ROOT / "scripts/video_size_reports/video_sizes.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/manifests"))
    parser.add_argument("--canvas", nargs=2, type=int, default=[896, 504], metavar=("W", "H"),
                        help="Canvas the pad_share column is measured against. The "
                             "shape_group column and the matched subset are "
                             "canvas-independent; only pad_share moves.")
    parser.add_argument("--match-ratio", type=int, default=4, metavar="K",
                        help="Fakes kept per real inside each resolution of the "
                             "matched subset. Larger K sharpens the fake side but "
                             "drops whole resolutions that cannot supply K, and "
                             "reals are the scarce class.")
    parser.add_argument("--min-wide-aspect", type=float, default=1.65,
                        help="Boundary between the square and wide clusters. The "
                             "corpus has no video between 1.60 and 1.70.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with open(args.manifest, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        base_fields = list(reader.fieldnames)
        rows = list(reader)
    fieldnames = base_fields + EXTRA_COLUMNS
    print("read {} rows from {}".format(len(rows), args.manifest))

    annotate(rows, load_sizes(args.sizes), tuple(args.canvas), args.min_wide_aspect)
    audit(rows, "BEFORE  (the manifest both pre-training runs used)")

    moved = drop_leaked_donors(rows)
    print("\ndonor leak repair: moved to {}".format(DONOR_BUCKET))
    for key in sorted(moved):
        print("  {}:{}  {} fakes".format(key[0], key[1], moved[key]))
    audit(rows, "AFTER donor repair")

    output = Path(args.output_dir)
    clean_path = output / "combined_manifest_supervised_clean.csv"
    print("")
    write([dict(row) for row in rows], clean_path, fieldnames)
    clean_counts = counts(rows)

    shapes = method_shapes(rows)
    report, dropped = match_on_resolution(rows, args.match_ratio, args.seed)
    print("\nresolution matching (ratio chosen per split, at most {}:1)".format(args.match_ratio))
    for key in sorted(report):
        value = report[key]
        if value.get("passed_through"):
            print("  {:<16} no shortcut present, left whole".format(key))
            continue
        print("  {:<16} wide candidates {:>5} -> {:>4} real + {:>4} fake at {}:1".format(
            key, value["wide_candidates"], value["matched_real"],
            value["matched_fake"], value["chosen_ratio"]))
    for key in sorted(dropped):
        print("  {}:{} moved {} rows to {}".format(key[0], key[1], dropped[key], SHAPE_BUCKET))
    audit(rows, "AFTER resolution matching  (0.5000 means the shortcut is gone)")

    unusable = sorted(name for name, shape in shapes.items() if not shape["wide"])
    if unusable:
        print("\n  {} of {} CelebDFv3 forgery methods emit no wide video at all, so no"
              "\n  shape-controlled comparison against a 94%-wide real class is possible"
              "\n  for them in this corpus. They are absent from the matched manifest:".format(
                  len(unusable), len(shapes)))
        for name in unusable:
            print("    {:<32} {} square videos".format(name, shapes[name]["square"]))

    matched_path = output / "combined_manifest_shape_matched.csv"
    print("")
    write(rows, matched_path, fieldnames)

    summary = {
        "built_from": str(args.manifest),
        "canvas": list(args.canvas),
        "match_ratio": args.match_ratio,
        "min_wide_aspect": args.min_wide_aspect,
        "seed": args.seed,
        "donor_repair": {"{}:{}".format(*k): v for k, v in sorted(moved.items())},
        "resolution_matching": report,
        "celebdfv3_method_shapes": shapes,
        "methods_without_wide_video": unusable,
        "clean_manifest_videos": clean_counts,
        "matched_manifest_videos": counts(rows),
    }
    summary_path = output / "combined_manifest_clean_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("  wrote {}".format(summary_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
