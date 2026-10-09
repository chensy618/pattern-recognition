"""
Performance metrics, implemented from scratch.

All metrics refer to the positive class (breach = 1), because the goal is to
identify the sites that are likely to breach the sea-lice limit:

    Precision = TP / (TP + FP)      fraction of flagged sites that do breach
    Recall    = TP / (TP + FN)      fraction of breaching sites that are flagged
    F1        = 2 P R / (P + R)     harmonic mean of precision and recall

Undefined ratios (0/0, e.g. precision when nothing is predicted positive) are
set to 0, which is the usual convention.
"""

import numpy as np


def confusion_counts(y_true, y_pred):
    """Return (TP, FP, FN, TN) for binary labels where 1 = breacher."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def precision_recall_f1(y_true, y_pred):
    """Precision, recall and F1 of the positive class."""
    tp, fp, fn, _ = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def best_f1_threshold(y_true, scores, n_candidates=200):
    """
    Decision threshold on a continuous score that maximises F1.

    Candidate thresholds are quantiles of the observed scores, so the search
    adapts to the scale of the score (probability or linear output).
    Must only be called on validation data, never on the test set.
    Returns (threshold, best_f1).
    """
    scores = np.asarray(scores, dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_candidates)))
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = precision_recall_f1(y_true, (scores >= t).astype(int))[2]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), best_f1
