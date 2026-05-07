from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


EPS = 1e-7



def choose_threshold_f1(y: np.ndarray, p: np.ndarray) -> float:
    """
    Learn threshold on validation set only.
    Exactly the user's requested rule.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    opt_thres = 0.0
    opt_f1 = -1.0
    for i in np.arange(0.001, 0.999, 0.001):
        cur = f1_score(y, p >= i)
        if cur >= opt_f1:
            opt_thres = float(i)
            opt_f1 = float(cur)
    return opt_thres



def _safe_logloss(y: np.ndarray, p: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=int)
    try:
        return float(log_loss(y, p, sample_weight=sample_weight, labels=[0, 1]))
    except Exception:
        return float("nan")



def _safe_auroc(y: np.ndarray, p: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p, sample_weight=sample_weight))
    except Exception:
        return float("nan")



def _safe_auprc(y: np.ndarray, p: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, p, sample_weight=sample_weight))
    except Exception:
        return float("nan")



def _weighted_binary_counts(
    y: np.ndarray,
    yhat: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    if sample_weight is None:
        sample_weight = np.ones_like(y, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    tp = float(sample_weight[(y == 1) & (yhat == 1)].sum())
    fp = float(sample_weight[(y == 0) & (yhat == 1)].sum())
    tn = float(sample_weight[(y == 0) & (yhat == 0)].sum())
    fn = float(sample_weight[(y == 1) & (yhat == 0)].sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}



def compute_binary_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    yhat = (prob >= threshold).astype(int)

    counts = _weighted_binary_counts(y_true, yhat, sample_weight)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    out: Dict[str, float] = {}
    out["logloss"] = _safe_logloss(y_true, prob, sample_weight)
    if sample_weight is None:
        out["brier"] = float(np.mean((prob - y_true) ** 2))
    else:
        sw = np.asarray(sample_weight, dtype=float)
        out["brier"] = float(np.average((prob - y_true) ** 2, weights=sw))
    out["auroc"] = _safe_auroc(y_true, prob, sample_weight)
    out["auprc"] = _safe_auprc(y_true, prob, sample_weight)
    try:
        out["f1"] = float(f1_score(y_true, yhat, sample_weight=sample_weight))
    except Exception:
        out["f1"] = float("nan")

    pos = tp + fn
    out["tpr"] = float(tp / pos) if pos > 0 else float("nan")
    out["fnr"] = float(fn / pos) if pos > 0 else float("nan")
    return out



def metric_names_default() -> List[str]:
    # return ["logloss", "brier", "auroc", "auprc", "f1", "fnr", "tpr"]

    return ["logloss"]
