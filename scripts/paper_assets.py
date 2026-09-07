"""Build the two assets the paper's main results rest on.

The table reports validation AUROC as best *and* run mean, because the best is
the quantity the checkpoint was selected on and the mean is what the run
actually looked like. On DFD they differ by 0.16 while the mean itself sits
below chance, and a table carrying only the best invites the reader to conclude
that Stage A worked there.

The figure makes the same point without prose: every epoch's validation reading
as a strip, the selected epoch marked on it, and the held-out test result
beside it. Stage A and Stage B sit in separate columns because they are
different heads on the same extractor -- a line drawn between them would imply
a progression that was never measured.

Usage:
    python scripts/paper_assets.py --output-dir results/paper
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

RUNS = (
    ("DFD", "Stage A", "results/run_stage_a_dfd_decimate_full60/history.json", None),
    ("DFD", "Stage B", "results/run_stage_a_dfd_decimate_stageb_full60/history.json",
     "results/run_stage_a_dfd_decimate/reports/test.json"),
    ("CelebDF++", "Stage A", "results/run_stage_a_celebdfv3_face_pretrain/history.json", None),
    ("CelebDF++", "Stage B", "results/run_stage_a_celebdfv3_face_stageb_celeb/history.json",
     "results/run_stage_a_celebdfv3_face_stageb_celeb/report_test.json"),
)

INK = "#14171B"
MUTED = "#565E68"
FAINT = "#9AA2AC"
VALIDATION = "#2a78d6"
TEST = "#eb6834"


def read(path):
    history = json.load(open(ROOT / path, encoding="utf-8"))
    return [item["selection_value"] for item in history
            if item.get("selection_value") is not None]


def collect():
    rows = []
    for dataset, stage, history_path, test_path in RUNS:
        values = read(history_path)
        test = None
        if test_path:
            test = json.load(open(ROOT / test_path, encoding="utf-8"))["auroc"]
        rows.append({
            "dataset": dataset, "stage": stage, "epochs": len(values),
            "values": values, "best": max(values),
            "best_epoch": values.index(max(values)) + 1,
            "mean": sum(values) / len(values), "test": test,
        })
    return rows


def write_table(rows, destination):
    """LNCS-ready booktabs, one row per dataset."""
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["stage"]] = row
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Validation and held-out test AUROC under the "
        r"target-and-donor-disjoint protocol. Validation is reported as the best "
        r"epoch, which is the checkpoint selection criterion, and as the mean over "
        r"the run. Each configuration was trained with a single seed.}",
        r"\label{tab:main-results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Stage A validation} & "
        r"\multicolumn{2}{c}{Stage B validation} & Held-out \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Dataset & best & mean & best & mean & test \\",
        r"\midrule",
    ]
    for dataset in ("DFD", "CelebDF++"):
        a, b = by_dataset[dataset]["Stage A"], by_dataset[dataset]["Stage B"]
        lines.append("{} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
            dataset, a["best"], a["mean"], b["best"], b["mean"], b["test"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    destination.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", destination)


def write_figure(rows, destination_stem):
    figure, axes = plt.subplots(figsize=(6.0, 3.2), dpi=200)
    positions = {("DFD", "Stage A"): 0.0, ("DFD", "Stage B"): 0.9,
                 ("CelebDF++", "Stage A"): 2.3, ("CelebDF++", "Stage B"): 3.2}
    rng = np.random.default_rng(7)

    axes.axhline(0.5, color=TEST, linestyle="--", linewidth=1.0, zorder=1)
    axes.text(-0.40, 0.504, "chance", color=TEST, fontsize=7.5, va="bottom", ha="left")

    for row in rows:
        x = positions[(row["dataset"], row["stage"])]
        values = np.asarray(row["values"])
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        axes.scatter(x + jitter, values, s=9, color=VALIDATION, alpha=0.35,
                     linewidths=0, zorder=2)
        axes.scatter([x], [row["best"]], s=46, marker="D", facecolor="white",
                     edgecolor=VALIDATION, linewidths=1.4, zorder=4)
        axes.hlines(row["mean"], x - 0.26, x + 0.26, color=MUTED,
                    linewidth=1.2, zorder=3)
        if row["test"] is not None:
            axes.scatter([x + 0.52], [row["test"]], s=54, marker="s",
                         color=TEST, zorder=5)
            axes.annotate("", xy=(x + 0.47, row["test"]), xytext=(x + 0.10, row["best"]),
                          arrowprops=dict(arrowstyle="->", color=FAINT,
                                          linewidth=1.0, shrinkA=6, shrinkB=6),
                          zorder=3)
            axes.text(x + 0.64, row["test"], "{:.4f}".format(row["test"]),
                      color=TEST, fontsize=8, ha="left", va="center")

    for (dataset, stage), x in positions.items():
        axes.text(x, 0.312, stage, fontsize=8, color=MUTED, ha="center")
    axes.text(0.45, 0.286, "DFD, whole frame", fontsize=8.5, color=INK, ha="center")
    axes.text(2.75, 0.286, "CelebDF++, face crop", fontsize=8.5, color=INK, ha="center")

    axes.set_ylabel("Video-level AUROC", fontsize=9)
    axes.set_ylim(0.27, 0.74)
    axes.set_xlim(-0.45, 4.35)
    axes.set_xticks([])
    axes.set_yticks(np.arange(0.35, 0.71, 0.05))
    axes.tick_params(axis="y", labelsize=8, colors=MUTED)
    for side in ("top", "right", "bottom"):
        axes.spines[side].set_visible(False)
    axes.spines["left"].set_color(FAINT)
    axes.grid(axis="y", color="#E4E7EB", linewidth=0.7, zorder=0)
    axes.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=VALIDATION, alpha=0.45,
                   markersize=4, label="validation, one point per epoch"),
        plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
                   markeredgecolor=VALIDATION, markersize=6, label="selected epoch (best)"),
        plt.Line2D([], [], color=MUTED, linewidth=1.2, label="run mean"),
        plt.Line2D([], [], marker="s", linestyle="", color=TEST, markersize=6,
                   label="held-out test"),
    ]
    axes.legend(handles=handles, fontsize=7.5, loc="upper left", frameon=False,
                handletextpad=0.5, borderaxespad=0.2, labelcolor=MUTED)
    figure.tight_layout()
    for suffix in (".pdf", ".png"):
        figure.savefig(str(destination_stem) + suffix, bbox_inches="tight",
                       transparent=False, facecolor="white")
        print("wrote", str(destination_stem) + suffix)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(ROOT / "results/paper"))
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rows = collect()
    print("{:<11}{:<9}{:>8}{:>9}{:>9}{:>12}{:>9}".format(
        "dataset", "stage", "epochs", "best", "mean", "best epoch", "test"))
    for row in rows:
        print("{:<11}{:<9}{:>8}{:>9.4f}{:>9.4f}{:>12}{:>9}".format(
            row["dataset"], row["stage"], row["epochs"], row["best"], row["mean"],
            row["best_epoch"],
            "{:.4f}".format(row["test"]) if row["test"] is not None else "-"))
    print()
    write_table(rows, destination / "main_results.tex")
    write_figure(rows, destination / "val_test_gap")
    with open(destination / "main_results.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print("wrote", destination / "main_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
