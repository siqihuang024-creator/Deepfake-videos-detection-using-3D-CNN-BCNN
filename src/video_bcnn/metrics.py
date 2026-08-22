"""Threshold calibration and real-vs-fake detection metrics."""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def calibrate_threshold(real_anomaly_scores, false_positive_rate):
    scores = np.asarray(real_anomaly_scores, dtype=np.float64)
    if not len(scores):
        raise ValueError("At least one real validation score is required for calibration.")
    return float(np.quantile(scores, 1.0 - float(false_positive_rate)))


def detection_metrics(real_labels, anomaly_scores, threshold, target_fpr=0.05):
    labels = np.asarray(real_labels, dtype=np.int64)
    scores = np.asarray(anomaly_scores, dtype=np.float64)
    fake_labels = 1 - labels
    predicted_fake = (scores >= float(threshold)).astype(np.int64)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(fake_labels, predicted_fake)),
        "balanced_accuracy": float(balanced_accuracy_score(fake_labels, predicted_fake)),
        "confusion_matrix": confusion_matrix(fake_labels, predicted_fake, labels=[0, 1]).tolist(),
        "observed_false_positive_rate": float(predicted_fake[fake_labels == 0].mean()),
        "observed_true_positive_rate": float(predicted_fake[fake_labels == 1].mean()),
    }
    if np.unique(fake_labels).size < 2:
        return dict(result, auroc=math.nan, average_precision=math.nan, eer=math.nan)
    result["auroc"] = float(roc_auc_score(fake_labels, scores))
    result["average_precision"] = float(average_precision_score(fake_labels, scores))
    fpr, tpr, _ = roc_curve(fake_labels, scores)
    result["target_false_positive_rate"] = float(target_fpr)
    result["tpr_at_target_fpr"] = float(np.interp(float(target_fpr), fpr, tpr))
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    result["eer"] = float((fpr[index] + fnr[index]) / 2.0)
    return result
