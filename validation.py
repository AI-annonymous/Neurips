from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .cmi import CMIConfig, DiscreteCMIConfig, estimate_cmi_stage1, estimate_cmi_stage1_discrete, estimate_cmi_stage2, estimate_cmi_stage2_discrete
from .data import OneHotBlockEncoder, build_input_matrix
from .metrics import compute_binary_metrics, metric_names_default
from .models import PosteriorConfig, fit_temperature_scaler, predict_proba, train_posterior_model
from .weight_diagnostics import (
    compute_weight_diagnostics, trimmed_t_a, diagnostic_flags,
)

# Metrics where larger values indicate worse performance.
LARGER_IS_WORSE = {"logloss", "brier", "fnr"}
# Metrics where smaller values indicate worse performance.
SMALLER_IS_WORSE = {"auroc", "auprc", "f1", "tpr"}


@dataclass
class WeightModelConfig:
    posterior: PosteriorConfig = field(default_factory=lambda: PosteriorConfig(max_epochs=80, patience=8))
    calibration_fraction: float = 0.2
    random_state: int = 123
    n_bootstrap: int = 1000
    bootstrap_alpha: float = 0.05
    bootstrap_seed: int = 123
    n_crossfit_folds: int = 5


@dataclass
class ResidualValidationResult:
    stage1_mi: float
    stage1_ce0: float
    stage1_ce1: float
    stage2_mi: float
    stage2_ce0: float
    stage2_ce1: float



def empty_residual_validation_result() -> ResidualValidationResult:
    return ResidualValidationResult(
        stage1_mi=float("nan"),
        stage1_ce0=float("nan"),
        stage1_ce1=float("nan"),
        stage2_mi=float("nan"),
        stage2_ce0=float("nan"),
        stage2_ce1=float("nan"),
    )



def _class_mapping_from_series(series: pd.Series):
    vals = list(pd.Series(series).dropna().astype(int if pd.api.types.is_numeric_dtype(series) else str).drop_duplicates())
    if pd.api.types.is_numeric_dtype(series):
        vals = sorted(vals)
    mapping = {v: i for i, v in enumerate(vals)}
    inverse = {i: v for v, i in mapping.items()}
    return mapping, inverse



def _map_with_mapping(series: pd.Series, mapping: Dict) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.Series(series).fillna(-1).astype(int).tolist()
    else:
        vals = pd.Series(series).fillna("__MISSING__").astype(str).tolist()
    default = next(iter(mapping.values()))
    return np.array([mapping.get(v, default) for v in vals], dtype=np.int64)



def is_metric_worse_than_overall(metric: str, metric_group: float, metric_overall: float, atol: float = 1e-12) -> bool:
    if not np.isfinite(metric_group) or not np.isfinite(metric_overall):
        return False
    if metric in LARGER_IS_WORSE:
        return bool(metric_group > metric_overall + atol)
    if metric in SMALLER_IS_WORSE:
        return bool(metric_group < metric_overall - atol)
    raise KeyError(f"Unknown metric direction for metric={metric}")



def build_metric_screen_table(
    df_test: pd.DataFrame,
    a_col: str,
    y_col: str,
    prob_col: str,
    threshold: float,
) -> pd.DataFrame:
    metrics = metric_names_default()
    mapping, inverse = _class_mapping_from_series(df_test[a_col])
    mapping_test = _map_with_mapping(df_test[a_col], mapping)

    y_true = df_test[y_col].astype(int).values
    p = df_test[prob_col].astype(float).values
    raw_all = compute_binary_metrics(y_true, p, threshold=threshold, sample_weight=None)

    rows: List[Dict] = []
    for class_idx, class_value in inverse.items():
        mask = mapping_test == class_idx
        n_group = int(mask.sum())
        n_pos_group = int(y_true[mask].sum()) if n_group > 0 else 0
        prev_group = float(n_pos_group / n_group) if n_group > 0 else float("nan")
        subgroup = f"{a_col}={class_value}"
        subgroup_metrics = compute_binary_metrics(y_true[mask], p[mask], threshold=threshold, sample_weight=None)
        for metric in metrics:
            metric_group = subgroup_metrics[metric]
            metric_overall = raw_all[metric]
            rows.append({
                "subgroup": subgroup,
                "group": class_value,
                "metric": metric,
                "threshold": float(threshold),
                "A_col": a_col,
                "n_group": n_group,
                "n_pos_group": n_pos_group,
                "prev_group": prev_group,
                "metric_overall": metric_overall,
                "metric_group": metric_group,
                "delta_overall": metric_group - metric_overall if np.isfinite(metric_group) and np.isfinite(metric_overall) else float("nan"),
                "ok": bool(is_metric_worse_than_overall(metric, metric_group, metric_overall)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Algorithm 2 (D'Amour et al.): cross-fitted weight predictions
# ---------------------------------------------------------------------------

def _crossfit_weight_predictions(
    df: pd.DataFrame,
    control_cols: Sequence[str],
    encoder: OneHotBlockEncoder,
    a_col: str,
    y_col: str,
    prob_col: str,
    device,
    config: WeightModelConfig,
) -> Tuple[np.ndarray, Dict, Dict]:
    """Implements Algorithm 2 from D'Amour et al.

    K-fold cross-fits the weight model P(A=a|V) on ``df`` itself so that
    every sample receives a held-out predicted weight.  Within each fold the
    training portion is further split into train / calibration for temperature
    scaling.

    Returns
    -------
    probs : ndarray, shape (N, n_classes)
        Cross-fitted P(A=a|V_i) for every sample.
    mapping, inverse : dicts
        Class label <-> index maps.
    """
    mapping, inverse = _class_mapping_from_series(df[a_col])
    y = _map_with_mapping(df[a_col], mapping)
    num_classes = len(mapping)

    include_y = y_col in control_cols
    include_r = prob_col in control_cols
    meta_cols = [c for c in control_cols if c not in {y_col, prob_col}]

    X = build_input_matrix(
        df, encoder, meta_cols,
        include_y=include_y, y_col=y_col,
        include_r=include_r, prob_col=prob_col,
    )

    n_folds = config.n_crossfit_folds
    probs_all = np.zeros((len(df), num_classes), dtype=np.float64)

    # Clamp n_folds to feasible range based on smallest class count
    counts = pd.Series(y).value_counts()
    min_class = int(counts.min()) if len(counts) else 0
    n_folds = min(n_folds, max(2, min_class))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.random_state)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), y)):
        y_tr_fold = y[train_idx]

        # Inner train / calibration split for temperature scaling
        cal_frac = config.calibration_fraction
        inner_min_class = int(pd.Series(y_tr_fold).value_counts().min())
        if inner_min_class < 2 or len(train_idx) < 10:
            # Not enough data for inner split — skip calibration
            model = train_posterior_model(
                X[train_idx], y_tr_fold, X[train_idx], y_tr_fold,
                num_classes=num_classes, device=device, config=config.posterior,
            )
            scaler = None
        else:
            inner_splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=cal_frac,
                random_state=config.random_state + fold_idx,
            )
            inner_tr_rel, inner_cal_rel = next(
                inner_splitter.split(np.arange(len(train_idx)), y_tr_fold)
            )
            actual_tr = train_idx[inner_tr_rel]
            actual_cal = train_idx[inner_cal_rel]

            model = train_posterior_model(
                X[actual_tr], y[actual_tr], X[actual_cal], y[actual_cal],
                num_classes=num_classes, device=device, config=config.posterior,
            )
            scaler = fit_temperature_scaler(model, X[actual_cal], y[actual_cal], device=device)

        probs_all[test_idx] = predict_proba(model, X[test_idx], device=device, scaler=scaler)

    return probs_all, mapping, inverse


# ---------------------------------------------------------------------------
# Legacy single-split weight model (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def _fit_weight_model(
    df_cal: pd.DataFrame,
    control_cols: Sequence[str],
    encoder: OneHotBlockEncoder,
    a_col: str,
    y_col: str,
    prob_col: str,
    device,
    config: WeightModelConfig,
):
    mapping, inverse = _class_mapping_from_series(df_cal[a_col])
    y = _map_with_mapping(df_cal[a_col], mapping)

    include_y = y_col in control_cols
    include_r = prob_col in control_cols
    meta_cols = [c for c in control_cols if c not in {y_col, prob_col}]

    X = build_input_matrix(df_cal, encoder, meta_cols, include_y=include_y, y_col=y_col, include_r=include_r, prob_col=prob_col)
    idx = np.arange(len(df_cal))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.calibration_fraction, random_state=config.random_state)
    tr_idx, cal_idx = next(splitter.split(idx, y))

    model = train_posterior_model(
        X[tr_idx], y[tr_idx], X[cal_idx], y[cal_idx],
        num_classes=len(mapping), device=device, config=config.posterior,
    )
    scaler = fit_temperature_scaler(model, X[cal_idx], y[cal_idx], device=device)

    return model, scaler, mapping, inverse, meta_cols, include_y, include_r



def _predict_weight_model(
    df: pd.DataFrame,
    model,
    scaler,
    mapping: Dict,
    inverse: Dict,
    encoder: OneHotBlockEncoder,
    meta_cols: Sequence[str],
    y_col: str,
    prob_col: str,
    include_y: bool,
    include_r: bool,
    device,
):
    X = build_input_matrix(df, encoder, meta_cols, include_y=include_y, y_col=y_col, include_r=include_r, prob_col=prob_col)
    probs = predict_proba(model, X, device=device, scaler=scaler)
    return probs


# ---------------------------------------------------------------------------
# Bootstrap CI for T_a
# ---------------------------------------------------------------------------

def _bootstrap_t_a(
    y_true: np.ndarray,
    p: np.ndarray,
    weights: np.ndarray,
    subgroup_mask: np.ndarray,
    metric: str,
    threshold: float,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float, float, bool]:
    """Percentile bootstrap CI for T_a.

    Following D'Amour et al.: un-normalized sample weights are treated as
    fixed and sampled alongside the data elements.

    Returns (ci_lo, ci_hi, p_value_boot, ci_contains_zero).
    """
    if n_bootstrap <= 1:
        return float("nan"), float("nan"), float("nan"), True

    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: List[float] = []

    from tqdm import tqdm
    for _ in tqdm(range(n_bootstrap), desc=f'bootstrap {metric}', leave=False, unit='iter'):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = p[idx]
        w_b = weights[idx]          # fixed weights, sampled alongside data
        mask_b = subgroup_mask[idx]
        if mask_b.sum() == 0:
            continue
        # No per-resample normalization: metrics self-normalize via
        # Sigma w*m / Sigma w (np.average, sklearn sample_weight, ratio metrics).
        group_metric_b = compute_binary_metrics(
            y_b[mask_b], p_b[mask_b], threshold=threshold, sample_weight=None,
        ).get(metric, float("nan"))
        weighted_metric_b = compute_binary_metrics(
            y_b, p_b, threshold=threshold, sample_weight=w_b,
        ).get(metric, float("nan"))
        t_b = group_metric_b - weighted_metric_b
        if np.isfinite(t_b):
            draws.append(float(t_b))

    if not draws:
        return float("nan"), float("nan"), float("nan"), True

    arr = np.asarray(draws, dtype=float)
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    p_boot = float(2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0)))
    p_boot = float(min(max(p_boot, 0.0), 1.0))
    contains_zero = bool(lo <= 0.0 <= hi)
    return lo, hi, p_boot, contains_zero


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_control_set(
        df_cal: pd.DataFrame,
        df_test: pd.DataFrame,
        control_name: str,
        control_cols: Sequence[str],
        encoder: OneHotBlockEncoder,
        a_col: str,
        y_col: str,
        prob_col: str,
        threshold: float,
        device,
        weight_model_config: WeightModelConfig,
        metric_screen_df: Optional[pd.DataFrame] = None,
        use_crossfit: bool = False,
) -> pd.DataFrame:
    """Compute T_a for one control set.

    If ``use_crossfit`` is False (default), fits the weight model on
    ``df_cal`` and predicts on ``df_test``.  This preserves test-set
    integrity when train/val/test splits are used for the full pipeline.

    If ``use_crossfit`` is True, implements Algorithm 2 from D'Amour et al.:
    the weight model is K-fold cross-fitted on ``df_test`` so every sample
    gets a held-out weight.  Only use this when there is no separate
    held-out test set.

    The returned DataFrame has weight-diagnostic columns (ess_over_n,
    p99_weight, max_smd_*, positivity_frac_*, ece, T_a_trimmed) for each
    (subgroup, metric) row, and ``df_out.attrs["sidecar_artifacts"]`` carries
    per-(subgroup) plot artifacts (per-block SMD, reliability bins,
    normalized weight arrays).
    """
    metrics = metric_names_default()
    if metric_screen_df is None:
        metric_screen_df = build_metric_screen_table(
            df_test=df_test, a_col=a_col, y_col=y_col,
            prob_col=prob_col, threshold=threshold,
        )
    metric_screen_df = metric_screen_df.copy()
    gate_lookup = {
        (row["group"], row["metric"]): row for _, row in metric_screen_df.iterrows()
    }

    # ---- obtain P(A=a|V_i) for every test sample ----
    import logging, time
    _log = logging.getLogger(__name__)
    _log.info('  [%s] fitting weight model | V=%s | n_test=%d | crossfit=%s',
              control_name, list(control_cols), len(df_test), use_crossfit)
    t_weight = time.perf_counter()

    if use_crossfit:
        # Algorithm 2: cross-fit on df_test
        probs_test, mapping, inverse = _crossfit_weight_predictions(
            df_test, control_cols, encoder, a_col, y_col, prob_col,
            device, weight_model_config,
        )
    else:
        # Legacy: train on df_cal, predict on df_test
        model, scaler, mapping, inverse, meta_cols, include_y, include_r = _fit_weight_model(
            df_cal, control_cols, encoder, a_col, y_col, prob_col, device, weight_model_config
        )
        probs_test = _predict_weight_model(
            df_test, model, scaler, mapping, inverse, encoder, meta_cols,
            y_col, prob_col, include_y, include_r, device,
        )

    _log.info('  [%s] weight model fitted in %.1fs | n_classes=%d',
              control_name, time.perf_counter() - t_weight, len(mapping))

    y_true = df_test[y_col].astype(int).values
    p = df_test[prob_col].astype(float).values

    n_ok = int(metric_screen_df["ok"].sum()) if metric_screen_df is not None else 0
    _log.info('  [%s] computing T_a + bootstrap (%d iters) for %d flagged (subgroup, metric) pairs',
              control_name, weight_model_config.n_bootstrap, n_ok)

    # Block columns for SMD (drop y_col / prob_col since those are added
    # separately by build_input_matrix and have no encoder block specs).
    block_cols_for_smd = [c for c in control_cols if c not in {y_col, prob_col}]

    sidecar_artifacts: List[Dict] = []
    out_rows: List[Dict] = []
    mapping_test = _map_with_mapping(df_test[a_col], mapping)

    raw_all = compute_binary_metrics(y_true, p, threshold=threshold, sample_weight=None)

    # Empty-weight scalar row template -- used by the early-skip branch and
    # by metric rows that fail the gate. NaN-filled so downstream extractors
    # find every column even when diagnostics couldn't be computed.
    _NAN_DIAG = {
        "ess_over_n": float("nan"),
        "p99_weight": float("nan"),
        "p999_weight": float("nan"),
        "max_weight": float("nan"),
        "max_smd_before": float("nan"),
        "max_smd_after": float("nan"),
        "positivity_frac_0.05": float("nan"),
        "positivity_frac_0.10": float("nan"),
        "ece": float("nan"),
    }

    for class_idx, class_value in inverse.items():
        # Define subgroup-level descriptors up front so the early-skip
        # branch below can use them without NameError.
        mask = mapping_test == class_idx
        subgroup = f"{a_col}={class_value}"
        n_group = int(mask.sum())
        n_pos_group = int(y_true[mask].sum()) if n_group > 0 else 0
        prev_group = float(n_pos_group / n_group) if n_group > 0 else float("nan")

        subgroup_metrics = compute_binary_metrics(
            y_true[mask], p[mask], threshold=threshold, sample_weight=None,
        )

        # w_i = g(v_i) = P(A=a|V_i), un-normalized per Algorithm 1.
        # Division by P(A=a) cancels in the self-normalizing ratio
        # M_a = Sum w_i m_i / Sum w_i, so we keep raw P(A=a|V_i).
        weights = probs_test[:, class_idx].copy()
        # Floor to small positive value to prevent zero-sum weights.
        weights = np.clip(weights, 1e-12, None)
        w_sum = float(weights.sum())
        if w_sum <= 0 or not np.isfinite(w_sum):
            _log.warning('  [%s] skipping subgroup %s=%s: weight sum is zero or non-finite',
                         control_name, a_col, class_value)
            for metric in metrics:
                gate_row = gate_lookup.get((class_value, metric), None)
                metric_overall = raw_all[metric]
                metric_group = subgroup_metrics[metric]
                ok = (bool(gate_row["ok"]) if gate_row is not None
                      else bool(is_metric_worse_than_overall(metric, metric_group, metric_overall)))
                delta_overall = (metric_group - metric_overall
                                 if np.isfinite(metric_group) and np.isfinite(metric_overall)
                                 else float("nan"))
                out_rows.append({
                    "subgroup": subgroup, "group": class_value, "control": control_name,
                    "ok": ok, "metric": metric, "threshold": float(threshold),
                    "V_used": ";".join(control_cols), "A_col": a_col,
                    "n_group": n_group, "n_pos_group": n_pos_group, "prev_group": prev_group,
                    "metric_overall(m)": metric_overall, "metric_group(M_mean_a)": metric_group,
                    "metric_weighted(M_a)": float("nan"),
                    "delta_overall(M_mean_a - m)": delta_overall,
                    "T_a": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "p_value_boot": float("nan"), "ci_contains_zero": True,
                    "T_a_empty": float("nan"), "Delta_a_m": float("nan"), "n_bootstrap": 0,
                    "T_a_trimmed": float("nan"), "n_trimmed_kept": 0,
                    **_NAN_DIAG,
                })
            continue

        weighted_metrics = compute_binary_metrics(
            y_true, p, threshold=threshold, sample_weight=weights,
        )

        # ----- weight diagnostics (one shot per subgroup, reused per metric) -----
        diag = compute_weight_diagnostics(
            df_test=df_test,
            encoder=encoder,
            block_cols=block_cols_for_smd,
            weights=weights,
            probs_class=probs_test[:, class_idx],
            subgroup_mask=mask,
        )
        flags = diagnostic_flags(diag)

        # On-screen diagnostic summary for this (control, subgroup).
        print(f"[{control_name}] {a_col}={class_value} | "
              f"ess_over_n={diag.ess_over_n:.3f} | "
              f"p99_weight={diag.p99_weight:.2f} | "
              f"max_smd_after={diag.max_smd_after:.3f} | "
              f"positivity_frac_0.10={diag.positivity_frac_eps10:.3f}")
        _log.info(
            '  [%s] %s | ess_over_n=%.3f p99(w)=%.2f maxSMD_before=%.3f maxSMD_after=%.3f pos@0.10=%.3f ECE=%.4f%s',
            control_name, subgroup,
            diag.ess_over_n, diag.p99_weight,
            diag.max_smd_before, diag.max_smd_after,
            diag.positivity_frac_eps10, diag.ece,
            (' [FLAGS: ' + ','.join(k for k, v in flags.items() if v) + ']'
             if any(flags.values()) else ''),
        )

        # Stash sidecar artifacts -- this is what diagnostic_plots reads.
        sidecar_artifacts.append({
            "control": control_name,
            "subgroup": subgroup,
            "class_value": class_value,
            "smd_table": diag.smd_table,
            "reliability": diag.reliability,
            "weights_normalized": diag.normalized_weights,
        })

        # Pack diagnostic scalars into a dict so each metric row gets the
        # same diagnostic columns appended consistently.
        diag_cols = {
            "ess_over_n": diag.ess_over_n,
            "p99_weight": diag.p99_weight,
            "p999_weight": diag.p999_weight,
            "max_weight": diag.max_weight,
            "max_smd_before": diag.max_smd_before,
            "max_smd_after": diag.max_smd_after,
            "positivity_frac_0.05": diag.positivity_frac_eps05,
            "positivity_frac_0.10": diag.positivity_frac_eps10,
            "ece": diag.ece,
        }

        for metric in metrics:
            gate_row = gate_lookup.get((class_value, metric), None)
            metric_overall = raw_all[metric]
            metric_group = subgroup_metrics[metric]
            ok = (bool(gate_row["ok"]) if gate_row is not None
                  else bool(is_metric_worse_than_overall(metric, metric_group, metric_overall)))
            delta_overall = (metric_group - metric_overall
                             if np.isfinite(metric_group) and np.isfinite(metric_overall)
                             else float("nan"))

            if ok:
                metric_weighted = weighted_metrics[metric]
                t_a = (metric_group - metric_weighted
                       if np.isfinite(metric_group) and np.isfinite(metric_weighted)
                       else float("nan"))
                if np.isfinite(t_a):
                    ci_lo, ci_hi, p_value_boot, ci_contains_zero = _bootstrap_t_a(
                        y_true=y_true, p=p, weights=weights,
                        subgroup_mask=mask, metric=metric, threshold=threshold,
                        n_bootstrap=weight_model_config.n_bootstrap,
                        alpha=weight_model_config.bootstrap_alpha,
                        seed=(weight_model_config.bootstrap_seed
                              + int(class_idx) * 1000
                              + sum(ord(ch) for ch in metric)),
                    )
                    # Crump-style trimmed T_a sensitivity check.
                    t_a_trim, n_kept = trimmed_t_a(
                        y_true=y_true, p=p, weights=weights,
                        probs_class=probs_test[:, class_idx],
                        subgroup_mask=mask, metric=metric,
                        threshold=threshold, eps=0.10,
                    )
                else:
                    ci_lo = ci_hi = p_value_boot = float("nan")
                    ci_contains_zero = True
                    t_a_trim, n_kept = float("nan"), 0
            else:
                metric_weighted = float("nan")
                t_a = float("nan")
                ci_lo = float("nan")
                ci_hi = float("nan")
                p_value_boot = float("nan")
                ci_contains_zero = True
                t_a_trim, n_kept = float("nan"), 0

            row = {
                "subgroup": subgroup,
                "group": class_value,
                "control": control_name,
                "ok": ok,
                "metric": metric,
                "threshold": float(threshold),
                "V_used": ";".join(control_cols),
                "A_col": a_col,
                "n_group": n_group,
                "n_pos_group": n_pos_group,
                "prev_group": prev_group,
                "metric_overall(m)": metric_overall,
                "metric_group(M_mean_a)": metric_group,
                "metric_weighted(M_a)": metric_weighted,
                "delta_overall(M_mean_a - m)": delta_overall,
                "T_a": t_a,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_value_boot": p_value_boot,
                "ci_contains_zero": ci_contains_zero,
                "T_a_empty": float("nan"),
                "Delta_a_m": float("nan"),
                "n_bootstrap": int(weight_model_config.n_bootstrap) if ok else 0,
                "T_a_trimmed": t_a_trim,
                "n_trimmed_kept": int(n_kept),
                **diag_cols,
            }
            out_rows.append(row)

    df_out = pd.DataFrame(out_rows)
    df_out.attrs["sidecar_artifacts"] = sidecar_artifacts
    return df_out




def add_gap_reduction(df_metrics: pd.DataFrame, baseline_control_name: str = "empty") -> pd.DataFrame:
    df = df_metrics.copy()
    # Drop placeholder columns to avoid suffix conflicts on merge
    df = df.drop(columns=["T_a_empty", "Delta_a_m"], errors="ignore")
    baseline = df[df["control"] == baseline_control_name][["A_col", "group", "metric", "T_a"]]
    baseline = baseline.rename(columns={"T_a": "T_a_empty"})
    df = df.merge(baseline, on=["A_col", "group", "metric"], how="left")
    denom = df["T_a_empty"].abs().replace(0.0, np.nan)
    df["Delta_a_m"] = 1.0 - (df["T_a"].abs() / denom)
    return df



def stagewise_residual_validation(
    df_cal: pd.DataFrame,
    df_test: pd.DataFrame,
    encoder: OneHotBlockEncoder,
    a_col: str,
    y_col: str,
    prob_col: str,
    v_y: Sequence[str],
    v_r: Sequence[str],
    device,
    cmi_config: CMIConfig,
    stage1_exact_discrete: bool = False,
    stage1_laplace_alpha: float = 1.0,
    stage2_exact_discrete: bool = False,
    stage2_laplace_alpha: float = 1.0,
    prob_col_bin: Optional[str] = None,
) -> ResidualValidationResult:
    if stage1_exact_discrete:
        res_y = estimate_cmi_stage1_discrete(
            df_test.reset_index(drop=True),
            v_y,
            a_col=a_col,
            y_col=y_col,
            config=DiscreteCMIConfig(laplace_alpha=stage1_laplace_alpha),
        )
    else:
        res_y = estimate_cmi_stage1(df_test.reset_index(drop=True), v_y, encoder, a_col=a_col, y_col=y_col, device=device, config=cmi_config)
    if stage2_exact_discrete:
        if not prob_col_bin:
            raise ValueError("stage2_exact_discrete requires prob_col_bin.")
        res_r = estimate_cmi_stage2_discrete(
            df_test.reset_index(drop=True),
            v_r,
            a_col=a_col,
            y_col=y_col,
            prob_col_bin=prob_col_bin,
            config=DiscreteCMIConfig(laplace_alpha=stage2_laplace_alpha),
        )
    else:
        res_r = estimate_cmi_stage2(df_test.reset_index(drop=True), v_r, encoder, a_col=a_col, y_col=y_col, prob_col=prob_col, device=device, config=cmi_config)
    return ResidualValidationResult(
        stage1_mi=res_y.mi,
        stage1_ce0=res_y.ce0,
        stage1_ce1=res_y.ce1,
        stage2_mi=res_r.mi,
        stage2_ce0=res_r.ce0,
        stage2_ce1=res_r.ce1,
    )
