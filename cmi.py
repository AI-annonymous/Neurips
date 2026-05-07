from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .data import OneHotBlockEncoder, build_input_matrix
from .models import PosteriorConfig, fit_temperature_scaler, predict_proba, train_posterior_model
from .random_utils import set_global_seed


@dataclass
class CMIConfig:
    n_outer_folds: int = 3
    inner_calibration_frac: float = 0.2
    random_state: int = 42
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)


@dataclass
class DiscreteCMIConfig:
    laplace_alpha: float = 0.0


@dataclass
class CMIResult:
    mi: float
    ce0: float
    ce1: float
    fold_values: List[float]
    n_classes: int
    class_values: List



def build_class_mapping(series: pd.Series) -> Tuple[Dict, Dict]:
    vals = list(pd.Series(series).dropna().astype(int if pd.api.types.is_numeric_dtype(series) else str).drop_duplicates())
    if pd.api.types.is_numeric_dtype(series):
        vals = sorted(vals)
    mapping = {v: i for i, v in enumerate(vals)}
    inverse = {i: v for v, i in mapping.items()}
    return mapping, inverse



def map_classes(series: pd.Series, mapping: Dict) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.Series(series).fillna(-1).astype(int).tolist()
    else:
        vals = pd.Series(series).fillna("__MISSING__").astype(str).tolist()
    first = next(iter(mapping.values()))
    return np.array([mapping.get(v, first) for v in vals], dtype=np.int64)



def _coerce_discrete_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").fillna(-1).astype(int)
        return vals.astype(object)
    return pd.Series(series).fillna("__MISSING__").astype(str)



def _make_context_keys(df: pd.DataFrame, subset_cols: Sequence[str]) -> pd.Series:
    if len(subset_cols) == 0:
        return pd.Series([("__EMPTY__",)] * len(df), index=df.index, dtype=object)
    context_df = pd.DataFrame(index=df.index)
    for c in subset_cols:
        context_df[c] = _coerce_discrete_series(df[c])
    return pd.Series(list(context_df.itertuples(index=False, name=None)), index=df.index, dtype=object)



def _conditional_entropy_from_counts(counts: np.ndarray, alpha: float) -> float:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError('counts must have shape [n_contexts, n_classes]')
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    n_classes = counts.shape[1]
    ent = 0.0
    for row in counts:
        n_ctx = float(row.sum())
        if n_ctx <= 0.0:
            continue
        if alpha > 0.0:
            probs = (row + alpha) / (n_ctx + alpha * n_classes)
        else:
            probs = row / n_ctx
        probs = probs[probs > 0]
        ent_ctx = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
        ent += (n_ctx / total) * ent_ctx
    return float(ent)



def estimate_cmi_stage2_discrete(
    df: pd.DataFrame,
    subset_cols: Sequence[str],
    a_col: str,
    y_col: str,
    prob_col_bin: str,
    config: Optional[DiscreteCMIConfig] = None,
) -> CMIResult:
    """Closed-form Stage 2 CMI: I(A; R_bin | Y, V) via contingency tables.

    Requires R to have been binarized upstream (e.g., using
    choose_threshold_f1 on the validation set).  Uses Laplace smoothing.

    Identity: I(A; R | Y, V) = H(A | Y, V) - H(A | Y, V, R).
    """
    cfg = config or DiscreteCMIConfig()
    if cfg.laplace_alpha < 0:
        raise ValueError('laplace_alpha must be nonnegative')

    a_series = _coerce_discrete_series(df[a_col])
    if len(a_series) == 0:
        return CMIResult(mi=0.0, ce0=0.0, ce1=0.0, fold_values=[0.0], n_classes=0, class_values=[])

    raw_values = list(pd.unique(a_series))
    if all(isinstance(v, (int, np.integer)) for v in raw_values):
        raw_values = sorted(raw_values)
    mapping = {v: i for i, v in enumerate(raw_values)}
    inverse = {i: v for v, i in mapping.items()}

    a_idx = np.array([mapping[v] for v in a_series.tolist()], dtype=np.int64)
    y_vals = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=1)
    r_vals = pd.to_numeric(df[prob_col_bin], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=1)
    v_keys = _make_context_keys(df, subset_cols)

    tmp = pd.DataFrame({
        'A': a_idx,
        'Y': y_vals.astype(int),
        'R': r_vals.astype(int),
        'V': v_keys,
    })

    # H(A | Y, V)  -- conditioning set is (Y, V), counts indexed by that
    yv_counts = tmp.groupby(['Y', 'V', 'A']).size().unstack(fill_value=0)
    yv_counts = yv_counts.reindex(columns=list(range(len(mapping))), fill_value=0)
    ce0 = _conditional_entropy_from_counts(yv_counts.to_numpy(dtype=float), alpha=cfg.laplace_alpha)

    # H(A | Y, V, R)  -- conditioning set is (Y, V, R)
    yvr_counts = tmp.groupby(['Y', 'V', 'R', 'A']).size().unstack(fill_value=0)
    yvr_counts = yvr_counts.reindex(columns=list(range(len(mapping))), fill_value=0)
    ce1 = _conditional_entropy_from_counts(yvr_counts.to_numpy(dtype=float), alpha=cfg.laplace_alpha)

    mi = float(max(ce0 - ce1, 0.0))
    return CMIResult(
        mi=mi,
        ce0=float(ce0),
        ce1=float(ce1),
        fold_values=[mi],
        n_classes=len(mapping),
        class_values=[inverse[i] for i in range(len(mapping))],
    )




def estimate_cmi_stage1_discrete(
    df: pd.DataFrame,
    subset_cols: Sequence[str],
    a_col: str,
    y_col: str,
    config: Optional[DiscreteCMIConfig] = None,
) -> CMIResult:
    cfg = config or DiscreteCMIConfig()
    if cfg.laplace_alpha < 0:
        raise ValueError('laplace_alpha must be nonnegative')

    a_series = _coerce_discrete_series(df[a_col])
    if len(a_series) == 0:
        return CMIResult(mi=0.0, ce0=0.0, ce1=0.0, fold_values=[0.0], n_classes=0, class_values=[])

    raw_values = list(pd.unique(a_series))
    if all(isinstance(v, (int, np.integer)) for v in raw_values):
        raw_values = sorted(raw_values)
    mapping = {v: i for i, v in enumerate(raw_values)}
    inverse = {i: v for v, i in mapping.items()}

    a_idx = np.array([mapping[v] for v in a_series.tolist()], dtype=np.int64)
    y_vals = pd.to_numeric(df[y_col], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=1)
    v_keys = _make_context_keys(df, subset_cols)

    tmp = pd.DataFrame({
        'A': a_idx,
        'Y': y_vals.astype(int),
        'V': v_keys,
    })

    av_counts = tmp.groupby(['V', 'A']).size().unstack(fill_value=0)
    av_counts = av_counts.reindex(columns=list(range(len(mapping))), fill_value=0)
    ce0 = _conditional_entropy_from_counts(av_counts.to_numpy(dtype=float), alpha=cfg.laplace_alpha)

    ayv_counts = tmp.groupby(['Y', 'V', 'A']).size().unstack(fill_value=0)
    ayv_counts = ayv_counts.reindex(columns=list(range(len(mapping))), fill_value=0)
    ce1 = _conditional_entropy_from_counts(ayv_counts.to_numpy(dtype=float), alpha=cfg.laplace_alpha)

    mi = float(max(ce0 - ce1, 0.0))
    return CMIResult(
        mi=mi,
        ce0=float(ce0),
        ce1=float(ce1),
        fold_values=[mi],
        n_classes=len(mapping),
        class_values=[inverse[i] for i in range(len(mapping))],
    )



def _train_calibrated_pair(
    X0_train: np.ndarray,
    X1_train: np.ndarray,
    y_train: np.ndarray,
    X0_cal: np.ndarray,
    X1_cal: np.ndarray,
    y_cal: np.ndarray,
    num_classes: int,
    device,
    config: CMIConfig,
):
    if len(X0_train) >= 20 and np.unique(y_train).size > 1:
        val_frac = min(0.15, max(0.1, 8 / max(len(X0_train), 1)))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=config.random_state)
        idx = np.arange(len(y_train))
        tr_idx, es_idx = next(splitter.split(idx, y_train))
    else:
        tr_idx = np.arange(len(y_train))
        es_idx = np.arange(len(y_train))

    model0 = train_posterior_model(
        X0_train[tr_idx], y_train[tr_idx], X0_train[es_idx], y_train[es_idx],
        num_classes=num_classes, device=device, config=config.posterior
    )
    model1 = train_posterior_model(
        X1_train[tr_idx], y_train[tr_idx], X1_train[es_idx], y_train[es_idx],
        num_classes=num_classes, device=device, config=config.posterior
    )
    scaler0 = fit_temperature_scaler(model0, X0_cal, y_cal, device=device)
    scaler1 = fit_temperature_scaler(model1, X1_cal, y_cal, device=device)
    return model0, scaler0, model1, scaler1



def _estimate_cmi_generic(
    df: pd.DataFrame,
    a_col: str,
    build_X0: Callable[[pd.DataFrame], np.ndarray],
    build_X1: Callable[[pd.DataFrame], np.ndarray],
    device,
    config: CMIConfig,
) -> CMIResult:
    set_global_seed(config.random_state)
    mapping, inverse = build_class_mapping(df[a_col])
    y_all = map_classes(df[a_col], mapping)
    num_classes = len(mapping)

    counts = pd.Series(y_all).value_counts()
    min_class_count = int(counts.min()) if len(counts) else 0
    n_splits = min(config.n_outer_folds, max(2, min_class_count)) if min_class_count >= 2 else 0
    if n_splits < 2:
        raise ValueError(
            f'Need at least 2 examples in every subgroup class to estimate CMI; got min_class_count={min_class_count} for A={a_col}.'
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    idx_all = np.arange(len(df))
    fold_values: List[float] = []
    ce0_vals: List[float] = []
    ce1_vals: List[float] = []

    X0_all = build_X0(df)
    X1_all = build_X1(df)

    import logging, time
    _log = logging.getLogger(__name__)

    for fold_id, (tr_idx, te_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        t_fold = time.perf_counter()
        _log.info('  CMI fold %d/%d | n_train=%d n_test=%d (A=%s, X0_dim=%d, X1_dim=%d)',
                   fold_id, n_splits, len(tr_idx), len(te_idx), a_col,
                   X0_all.shape[1], X1_all.shape[1])
        y_tr = y_all[tr_idx]
        if len(tr_idx) < 10:
            continue
        inner_counts = pd.Series(y_tr).value_counts()
        inner_min_class = int(inner_counts.min()) if len(inner_counts) else 0
        if inner_min_class < 2:
            continue
        calib_frac = config.inner_calibration_frac
        max_calib_frac = min(0.5, max(0.1, 1.0 - (2.0 / max(len(tr_idx), 2))))
        calib_frac = min(calib_frac, max_calib_frac)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=calib_frac,
            random_state=config.random_state + fold_id,
        )
        inner_train_rel, inner_cal_rel = next(splitter.split(np.arange(len(tr_idx)), y_tr))
        inner_train_idx = tr_idx[inner_train_rel]
        inner_cal_idx = tr_idx[inner_cal_rel]

        model0, scaler0, model1, scaler1 = _train_calibrated_pair(
            X0_all[inner_train_idx],
            X1_all[inner_train_idx],
            y_all[inner_train_idx],
            X0_all[inner_cal_idx],
            X1_all[inner_cal_idx],
            y_all[inner_cal_idx],
            num_classes=num_classes,
            device=device,
            config=config,
        )

        p0 = np.clip(predict_proba(model0, X0_all[te_idx], device=device, scaler=scaler0), 1e-7, 1.0)
        p1 = np.clip(predict_proba(model1, X1_all[te_idx], device=device, scaler=scaler1), 1e-7, 1.0)
        yt = y_all[te_idx]

        ce0 = float(np.mean(-np.log(p0[np.arange(len(yt)), yt])))
        ce1 = float(np.mean(-np.log(p1[np.arange(len(yt)), yt])))
        mi = ce0 - ce1
        ce0_vals.append(ce0)
        ce1_vals.append(ce1)
        fold_values.append(mi)
        _log.info('  CMI fold %d/%d done | ce0=%.4f ce1=%.4f mi=%.4f | %.1fs',
                   fold_id, n_splits, ce0, ce1, mi, time.perf_counter() - t_fold)

    if not fold_values:
        raise RuntimeError(f'No valid outer folds were available when estimating CMI for A={a_col}.')

    return CMIResult(
        mi=float(np.mean(fold_values)),
        ce0=float(np.mean(ce0_vals)),
        ce1=float(np.mean(ce1_vals)),
        fold_values=fold_values,
        n_classes=num_classes,
        class_values=[inverse[i] for i in range(num_classes)],
    )



def estimate_cmi_stage1(
    df: pd.DataFrame,
    subset_cols: Sequence[str],
    encoder: OneHotBlockEncoder,
    a_col: str,
    y_col: str,
    device,
    config: CMIConfig,
) -> CMIResult:
    def build_X0(xdf: pd.DataFrame) -> np.ndarray:
        return build_input_matrix(xdf, encoder, subset_cols, include_y=False)

    def build_X1(xdf: pd.DataFrame) -> np.ndarray:
        return build_input_matrix(xdf, encoder, subset_cols, include_y=True, y_col=y_col)

    return _estimate_cmi_generic(df, a_col, build_X0, build_X1, device=device, config=config)



def estimate_cmi_stage2(
    df: pd.DataFrame,
    subset_cols: Sequence[str],
    encoder: OneHotBlockEncoder,
    a_col: str,
    y_col: str,
    prob_col: str,
    device,
    config: CMIConfig,
) -> CMIResult:
    def build_X0(xdf: pd.DataFrame) -> np.ndarray:
        return build_input_matrix(xdf, encoder, subset_cols, include_y=True, y_col=y_col)

    def build_X1(xdf: pd.DataFrame) -> np.ndarray:
        return build_input_matrix(xdf, encoder, subset_cols, include_y=True, y_col=y_col, include_r=True, prob_col=prob_col)

    return _estimate_cmi_generic(df, a_col, build_X0, build_X1, device=device, config=config)
