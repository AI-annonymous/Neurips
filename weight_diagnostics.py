"""Weight diagnostics for the D'Amour-style validation pipeline.

For each (subgroup, metric) row produced by ``evaluate_control_set``, this module
computes a suite of diagnostics that probe whether the gap closure
T_{a,m}(V*) is driven by genuine covariate adjustment or by a degenerate
weighting regime (extreme weights / poor positivity / miscalibration).

All functions are pure numpy / pandas; no torch. The intended call site is
inside ``evaluate_control_set`` after ``probs_test`` has been computed and the
per-class ``weights = probs_test[:, class_idx]`` are formed.

Math reference (one place, the rest of the file just implements):

    Hájek estimator       M_a(V*) = sum_i w_i m_i / sum_i w_i,    w_i = P̂(A=a|V_i*)
    Kish ESS              ESS = (sum w)^2 / sum w^2
    Normalized weights    w̃_i = n * w_i / sum_j w_j               (mean = 1)
    SMD (Austin 2009)     (mean_target - mean_proposal) / sqrt((var_target + var_proposal)/2)
    ECE                   sum_b (|S_b|/n) * |acc(S_b) - conf(S_b)|
    Trimmed T_a           drop samples with P̂(A=a|V*) outside (eps, 1-eps), recompute

The diagnostics row this module returns is designed to slot in as additional
columns on the existing T_a table. Plot-ready artifacts (per-block SMD,
weight histogram bins, reliability bins) are returned separately so the
table stays compact and a sidecar parquet/json holds the heavy detail.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import OneHotBlockEncoder
from .metrics import compute_binary_metrics


# ---------------------------------------------------------------------------
#  Primitive helpers
# ---------------------------------------------------------------------------

def kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size: (sum w)^2 / sum w^2.

    ESS / n in (0, 1]. ESS = n iff weights are constant. ESS -> 1 if a single
    sample carries all mass.
    """
    w = np.asarray(weights, dtype=float)
    s1 = w.sum()
    s2 = (w * w).sum()
    if s2 <= 0 or not np.isfinite(s2):
        return 0.0
    return float(s1 * s1 / s2)


def normalize_weights_mean_one(weights: np.ndarray) -> np.ndarray:
    """Rescale so weights sum to n (i.e., have empirical mean 1).

    With this normalization, w̃_i has the interpretation 'this sample is
    contributing w̃_i times more than an average sample'.
    """
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return np.zeros_like(w)
    return len(w) * w / s


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        return float("nan")
    return float((w * x).sum() / s)


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    s = float(w.sum())
    if s <= 0 or not np.isfinite(s):
        return float("nan")
    m = float((w * x).sum() / s)
    return float((w * (x - m) ** 2).sum() / s)


# ---------------------------------------------------------------------------
#  Standardized mean differences (Austin 2009)
# ---------------------------------------------------------------------------

def smd_per_dim(
    X_full: np.ndarray,           # (n, d) design matrix for the full test set
    subgroup_mask: np.ndarray,    # (n,) bool, True for samples in subgroup A=a
    weights: np.ndarray,          # (n,) w_i = P̂(A=a|V_i*)
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature SMD before and after weighting.

    Target distribution is the unweighted subgroup A=a. Proposal distribution
    is the full test set; before weighting it's unweighted, after weighting
    it's weighted by P̂(A=a|V*). A successful weight model makes the
    weighted full-pop look like the unweighted subgroup.

    Returns
    -------
    smd_before, smd_after : ndarray, shape (d,)
        Per-feature SMD. Convention from Austin (2009): |SMD| < 0.1 = balanced.
    """
    X = np.asarray(X_full, dtype=float)
    mask = np.asarray(subgroup_mask, dtype=bool)
    w = np.asarray(weights, dtype=float)

    if mask.sum() == 0:
        d = X.shape[1]
        nan_v = np.full(d, np.nan)
        return nan_v, nan_v

    X_target = X[mask]
    mean_target = X_target.mean(axis=0)
    var_target = X_target.var(axis=0, ddof=0)

    # Before: unweighted full-pop proposal
    mean_full = X.mean(axis=0)
    var_full = X.var(axis=0, ddof=0)

    # After: weighted full-pop proposal
    s_w = w.sum()
    if s_w <= 0 or not np.isfinite(s_w):
        mean_full_w = mean_full.copy()
        var_full_w = var_full.copy()
    else:
        mean_full_w = (w[:, None] * X).sum(axis=0) / s_w
        var_full_w = (w[:, None] * (X - mean_full_w) ** 2).sum(axis=0) / s_w

    pooled_before = np.sqrt((var_target + var_full) / 2.0)
    pooled_after = np.sqrt((var_target + var_full_w) / 2.0)

    # Avoid division by zero on constant columns (one-hot of cardinality 1, etc.)
    eps = 1e-12
    smd_before = np.where(
        pooled_before > eps,
        (mean_target - mean_full) / np.maximum(pooled_before, eps),
        0.0,
    )
    smd_after = np.where(
        pooled_after > eps,
        (mean_target - mean_full_w) / np.maximum(pooled_after, eps),
        0.0,
    )
    return smd_before, smd_after


def smd_per_block(
    df_test: pd.DataFrame,
    encoder: OneHotBlockEncoder,
    block_cols: Sequence[str],
    subgroup_mask: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """Per V*-variable (= per block) SMD before/after weighting.

    A 'block' is one semantic variable in V*. For categoricals the block has
    multiple one-hot columns; we report max |SMD| across columns of the block
    (Austin's recommendation for multi-level categorical balance) AND mean
    |SMD| as a robustness check.

    Returns a DataFrame with one row per block:
        block | n_dim | block_type | smd_before_max | smd_after_max | smd_after_mean
    """
    rows: List[Dict] = []
    col_offset = 0
    blocks = encoder.transform_blocks(df_test, block_cols)
    for col in block_cols:
        block = blocks[col].astype(float)              # (n, d_block)
        spec = encoder.block_specs[col]
        sb, sa = smd_per_dim(block, subgroup_mask, weights)
        rows.append({
            "block": col,
            "block_type": spec.block_type,
            "n_dim": int(block.shape[1]),
            "smd_before_max": float(np.nanmax(np.abs(sb))) if sb.size else float("nan"),
            "smd_after_max": float(np.nanmax(np.abs(sa))) if sa.size else float("nan"),
            "smd_after_mean": float(np.nanmean(np.abs(sa))) if sa.size else float("nan"),
        })
        col_offset += block.shape[1]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Calibration of the weight model
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, pd.DataFrame]:
    """ECE with equal-width bins on [0,1].

    Parameters
    ----------
    probs : (n,) float
        Predicted P̂(A=a|V_i*), one per sample.
    labels : (n,) {0,1}
        Indicator that A_i = a.
    n_bins : int
        Number of bins in the reliability decomposition.

    Returns
    -------
    ece : float
    reliability : DataFrame with bin_lo, bin_hi, bin_mid, count, conf, acc
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = len(probs)
    if n == 0:
        return float("nan"), pd.DataFrame()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append({"bin_lo": float(lo), "bin_hi": float(hi),
                         "bin_mid": float((lo + hi) / 2),
                         "count": 0, "conf": float("nan"), "acc": float("nan")})
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += (cnt / n) * abs(acc - conf)
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi),
                     "bin_mid": float((lo + hi) / 2),
                     "count": cnt, "conf": conf, "acc": acc})
    return float(ece), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Positivity / overlap
# ---------------------------------------------------------------------------

def positivity_violation_fraction(
    probs_class: np.ndarray,
    eps: float = 0.05,
) -> float:
    """Fraction of test samples with P̂(A=a|V*) outside (eps, 1-eps).

    Crump et al. (2009) recommend eps ∈ {0.05, 0.1} for trimming.
    """
    p = np.asarray(probs_class, dtype=float)
    if len(p) == 0:
        return float("nan")
    bad = (p <= eps) | (p >= 1.0 - eps)
    return float(bad.mean())


# ---------------------------------------------------------------------------
#  Trimmed T_a (Crump et al. 2009)
# ---------------------------------------------------------------------------

def trimmed_t_a(
    y_true: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    probs_class: np.ndarray,
    subgroup_mask: np.ndarray,
    metric: str,
    threshold: float,
    eps: float = 0.1,
    min_kept: int = 20,
) -> Tuple[float, int]:
    """Recompute T_a after trimming samples with extreme propensities.

    Drops samples with P̂(A=a|V*) outside (eps, 1-eps), then recomputes
    T_a = subgroup metric - weighted full-pop metric on the trimmed sample.

    A T_a that is robust to trimming is defensible. A T_a that changes
    substantially under trimming was driven by extreme-propensity samples
    and should be reported with that caveat.

    Returns
    -------
    t_a_trimmed : float
    n_kept : int
    """
    keep = (probs_class > eps) & (probs_class < 1.0 - eps)
    n_kept = int(keep.sum())
    if n_kept < min_kept:
        return float("nan"), n_kept

    y_t = y_true[keep]
    p_t = p[keep]
    w_t = weights[keep]
    mask_t = subgroup_mask[keep]
    if mask_t.sum() < 5 or (~mask_t).sum() < 5:
        return float("nan"), n_kept

    sub_metrics = compute_binary_metrics(
        y_t[mask_t], p_t[mask_t], threshold=threshold, sample_weight=None
    )
    weighted_metrics = compute_binary_metrics(
        y_t, p_t, threshold=threshold, sample_weight=w_t
    )
    if not np.isfinite(sub_metrics[metric]) or not np.isfinite(weighted_metrics[metric]):
        return float("nan"), n_kept
    return float(sub_metrics[metric] - weighted_metrics[metric]), n_kept


# ---------------------------------------------------------------------------
#  Public API: one call gives all scalars + plot artifacts
# ---------------------------------------------------------------------------

@dataclass
class WeightDiagnostics:
    # Scalar fields suitable for the main paper table.
    ess: float                     # Kish ESS
    ess_over_n: float              # ESS / n
    p99_weight: float              # 99th percentile of normalized weights (mean=1)
    p999_weight: float             # 99.9th percentile
    max_weight: float              # max normalized weight
    max_smd_before: float          # max |SMD| across V* dims, unweighted full-pop vs subgroup
    max_smd_after: float           # max |SMD| across V* dims, weighted full-pop vs subgroup
    positivity_frac_eps05: float   # frac with P̂(A=a|V*) outside (0.05, 0.95)
    positivity_frac_eps10: float   # frac with P̂(A=a|V*) outside (0.10, 0.90)
    ece: float                     # ECE of P̂(A=a|V*) on test set
    # Artifacts (per-block SMD, reliability table, normalized weight array).
    # Kept here for sidecar export rather than the main table.
    smd_table: pd.DataFrame = None
    reliability: pd.DataFrame = None
    normalized_weights: np.ndarray = None


def compute_weight_diagnostics(
    df_test: pd.DataFrame,
    encoder: OneHotBlockEncoder,
    block_cols: Sequence[str],
    weights: np.ndarray,
    probs_class: np.ndarray,
    subgroup_mask: np.ndarray,
    n_ece_bins: int = 15,
) -> WeightDiagnostics:
    """One-stop diagnostics for a single (subgroup, class) pair.

    Parameters
    ----------
    df_test : DataFrame
        The test set used to compute weights and metrics.
    encoder : OneHotBlockEncoder
        Same encoder used to fit the weight model. Provides per-V*-variable
        block structure for SMD aggregation.
    block_cols : sequence of str
        The selected V* block column names (i.e., ``control_cols``, with
        ``y_col`` and ``prob_col`` filtered out as they are added separately
        in build_input_matrix and don't have block specs).
    weights : (n,)
        w_i = P̂(A=a|V_i*) used in the Hájek estimator.
    probs_class : (n,)
        Same as ``weights`` for now (kept separate to make the API explicit:
        positivity diagnostics conceptually use the propensity, weights
        downstream use whatever transformation the validation step applied).
    subgroup_mask : (n,) bool
        True for samples with A_i = a.
    """
    w = np.asarray(weights, dtype=float)
    pc = np.asarray(probs_class, dtype=float)

    # ESS
    ess = kish_ess(w)
    ess_over_n = ess / len(w) if len(w) > 0 else float("nan")

    # Normalized weights and tail percentiles
    w_norm = normalize_weights_mean_one(w)
    p99 = float(np.percentile(w_norm, 99)) if len(w_norm) else float("nan")
    p999 = float(np.percentile(w_norm, 99.9)) if len(w_norm) else float("nan")
    w_max = float(w_norm.max()) if len(w_norm) else float("nan")

    # Per-block SMD; restrict to columns that have block specs (i.e., metadata)
    meta_cols = [c for c in block_cols if c in encoder.block_specs]
    if meta_cols:
        smd_tbl = smd_per_block(df_test, encoder, meta_cols, subgroup_mask, w)
        max_smd_before = float(np.nanmax(smd_tbl["smd_before_max"].values)) if len(smd_tbl) else float("nan")
        max_smd_after = float(np.nanmax(smd_tbl["smd_after_max"].values)) if len(smd_tbl) else float("nan")
    else:
        smd_tbl = pd.DataFrame()
        max_smd_before = float("nan")
        max_smd_after = float("nan")

    # Positivity
    pf_05 = positivity_violation_fraction(pc, eps=0.05)
    pf_10 = positivity_violation_fraction(pc, eps=0.10)

    # Calibration
    labels = subgroup_mask.astype(int)
    ece, reliability = expected_calibration_error(pc, labels, n_bins=n_ece_bins)

    return WeightDiagnostics(
        ess=ess,
        ess_over_n=ess_over_n,
        p99_weight=p99,
        p999_weight=p999,
        max_weight=w_max,
        max_smd_before=max_smd_before,
        max_smd_after=max_smd_after,
        positivity_frac_eps05=pf_05,
        positivity_frac_eps10=pf_10,
        ece=ece,
        smd_table=smd_tbl,
        reliability=reliability,
        normalized_weights=w_norm,
    )


# ---------------------------------------------------------------------------
#  Quick interpretation helpers (for log lines / sanity gates)
# ---------------------------------------------------------------------------

def diagnostic_flags(d: WeightDiagnostics) -> Dict[str, bool]:
    """Boolean flags following standard rules of thumb.

    These are *not* hard rejections, just signals to surface in logs and
    paper appendix. Healthy run: most flags False.
    """
    return {
        "low_ess":           bool(np.isfinite(d.ess_over_n) and d.ess_over_n < 0.10),
        "borderline_ess":    bool(np.isfinite(d.ess_over_n) and 0.10 <= d.ess_over_n < 0.30),
        "extreme_tail":      bool(np.isfinite(d.p99_weight) and d.p99_weight > 10.0),
        "imbalance":         bool(np.isfinite(d.max_smd_after) and d.max_smd_after > 0.10),
        "severe_imbalance":  bool(np.isfinite(d.max_smd_after) and d.max_smd_after > 0.25),
        "positivity_alarm":  bool(np.isfinite(d.positivity_frac_eps10) and d.positivity_frac_eps10 > 0.10),
        "miscalibrated":     bool(np.isfinite(d.ece) and d.ece > 0.05),
    }


# ---------------------------------------------------------------------------
#  Sidecar artifact export (for diagnostic_plots.make_all_appendix_figures)
# ---------------------------------------------------------------------------

def write_sidecar_artifacts(sidecar_artifacts, out_dir: str, tag: str = "diagnostics"):
    """Persist per-(control, subgroup) artifacts for appendix plots.

    Filename format uses double-underscore as the field separator so
    parsing is unambiguous when control names and column names both
    contain single underscores (e.g. ``searched_vy``, ``age_bin``):

        {out_dir}/{tag}__{control}__{subgroup}__smd.csv
        {out_dir}/{tag}__{control}__{subgroup}__reliability.csv
        {out_dir}/{tag}__{control}__{subgroup}__weights.npy

    where {subgroup} is e.g. ``A-0``, ``age_bin-0``, ``machine_id_bin-2``.
    These are the inputs to the Love plot, reliability curve, and weight
    violin plot in diagnostic_plots.py.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    for art in sidecar_artifacts:
        ctrl = str(art["control"]).replace("/", "_")
        sg = str(art["subgroup"]).replace("/", "_").replace("=", "-")
        prefix = f"{tag}__{ctrl}__{sg}"
        smd_tbl = art.get("smd_table")
        if smd_tbl is not None and len(smd_tbl) > 0:
            smd_tbl.to_csv(f"{out_dir}/{prefix}__smd.csv", index=False)
        rel = art.get("reliability")
        if rel is not None and len(rel) > 0:
            rel.to_csv(f"{out_dir}/{prefix}__reliability.csv", index=False)
        w = art.get("weights_normalized")
        if w is not None:
            np.save(f"{out_dir}/{prefix}__weights.npy", w)