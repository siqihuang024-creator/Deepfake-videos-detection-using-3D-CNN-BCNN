"""Persist training histories, scores, and compact diagnostic reports."""

import csv

import matplotlib

# Rented GPU hosts are headless. Select Agg before pyplot is imported so the
# first epoch's curve export cannot fail on a missing display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from .utils import ensure_dir, save_json


def save_history(history, output_dir):
    output_dir = ensure_dir(output_dir)
    save_json(output_dir / "history.json", history)
    fields = [
        "epoch", "train_loss", "learning_rate", "selection_value",
        "validation_accuracy", "validation_balanced_accuracy", "validation_auroc",
        "validation_macro_dataset_auroc", "validation_eer", "validation_tpr_at_target_fpr",
        "dfd_auroc", "celebdfv3_auroc", "embedding_variance_mean",
        # Saturation evidence: a stage whose pre-activations mostly exceed |4|
        # has stopped passing gradient, which separates a dead activation from
        # a badly scaled objective.
        "stage1_saturated", "stage2_saturated", "stage3_saturated", "stage3_output_std",
    ]
    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            validation = row["validation"]
            stages = row.get("posterior_diagnostics", {}).get("activation_stages", [])
            def stage(index, key):
                return stages[index][key] if len(stages) > index else None
            writer.writerow({
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "learning_rate": row["learning_rate"],
                "selection_value": row["selection_value"],
                "validation_accuracy": validation["accuracy"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "validation_auroc": validation["auroc"],
                "validation_macro_dataset_auroc": validation["macro_dataset_auroc"],
                "validation_eer": validation["eer"],
                "validation_tpr_at_target_fpr": validation["tpr_at_target_fpr"],
                "dfd_auroc": validation["per_dataset"].get("DFD", {}).get("auroc"),
                "celebdfv3_auroc": validation["per_dataset"].get("CelebDFv3", {}).get("auroc"),
                "embedding_variance_mean": validation["embedding_variance_mean"],
                "stage1_saturated": stage(0, "saturated_fraction"),
                "stage2_saturated": stage(1, "saturated_fraction"),
                "stage3_saturated": stage(2, "saturated_fraction"),
                "stage3_output_std": stage(2, "output_std"),
            })
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="negative ELBO")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["validation"]["auroc"] for row in history], label="overall AUROC")
    plt.plot(epochs, [row["validation"]["macro_dataset_auroc"] for row in history], label="macro dataset AUROC")
    plt.xlabel("Epoch")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


def save_scores(values, output_path):
    fields = [
        "dataset", "path", "method", "target_id", "donor_id", "label_real",
        "anomaly_score", "predictive_mean", "predictive_std", "embedding_norm", "fps",
        "face_any_miss", "face_miss_fraction", "center_x_jitter", "center_y_jitter",
        "width_jitter", "height_jitter",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(values["labels"])):
            writer.writerow({
                "dataset": values["datasets"][index],
                "path": values["paths"][index],
                "method": values["methods"][index],
                "target_id": values["target_ids"][index],
                "donor_id": values["donor_ids"][index],
                "label_real": int(values["labels"][index]),
                "anomaly_score": float(values["scores"][index]),
                "predictive_mean": float(values["means"][index]),
                "predictive_std": float(values["stds"][index]),
                "embedding_norm": float(values["embedding_norms"][index]),
                "fps": float(values["fps"][index]),
                "face_any_miss": float(values["face_any_miss"][index]),
                "face_miss_fraction": float(values["face_miss_fraction"][index]),
                "center_x_jitter": float(values["center_x_jitter"][index]),
                "center_y_jitter": float(values["center_y_jitter"][index]),
                "width_jitter": float(values["width_jitter"][index]),
                "height_jitter": float(values["height_jitter"][index]),
            })


def save_evaluation_report(values, metrics, output_dir, split):
    output_dir = ensure_dir(output_dir)
    save_json(output_dir / "{}.json".format(split), metrics)
    save_scores(values, output_dir / "{}_scores.csv".format(split))
    labels, scores = values["labels"], values["scores"]
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.hist(scores[labels == 1], bins=30, alpha=0.7, label="real")
    plt.hist(scores[labels == 0], bins=30, alpha=0.7, label="fake")
    plt.axvline(metrics["threshold"], color="black", linestyle="--", label="threshold")
    plt.xlabel("Video anomaly score")
    plt.legend()
    plt.subplot(1, 2, 2)
    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(1 - labels, scores)
        plt.plot(fpr, tpr, label="AUROC {:.4f}".format(metrics["auroc"]))
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(output_dir / "{}_diagnostics.png".format(split), dpi=160)
    plt.close()
    return output_dir / "{}.json".format(split)
