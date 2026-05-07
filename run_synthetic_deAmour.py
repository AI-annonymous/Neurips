"""
Synthetic experiments for the Graph-Constrained Two-Stage Search method.

Causal graph (matches main paper Fig. 1):

        W_pre ───►  Y
          │          │
          │          ▼
        A_up ─►   X (measurement)  ◄─── W_acq
          │          │
          │          ▼
          ▲          R = f̂(X)
          │
        W_acq

Variables:
  - W_pre: pre-treatment confounders, affect Y AND X
  - W_acq: acquisition-stage confounders, affect X only (NOT Y)
  - Y:     binary label
  - X:     univariate measurement, X = h(Y, W_acq) + noise   (W_pre's
           effect on X is fully mediated by Y; no direct W_pre→X edge)
  - R:     classifier prediction R = f̂(X)  --  trained on (X_train, Y_train)
  - A:     subgroup membership; correlated with W_pre and W_acq through
           bidirected (unobserved) confounding

Faithful to D'Amour et al. (NeurIPS 2025) synthetic protocol:
  - X is univariate
  - R is the prediction of a classifier trained on (X, Y) on the training
    set (HistGradientBoostingClassifier or LogisticRegression)
  - We sample 70k, split 50k train / 20k held-out for fitting, but the
    full DGP also produces our val + test splits for validation and CMI
    estimation

Experiment 1 — Oracle graph validation (no search):
    Verify that conditioning on TRUE causal controls drives residual CMI
    and T_a to zero on held-out data.

Experiment 2 — Search recovery WITHOUT distractors (clean):
    Recover V_Y* and V_R* over a clean candidate pool (only true vars).

Experiment 3 — Search recovery WITH distractors:
    Same but adds D_pre / D_acq distractors and the W_post variable T1.

In all experiments, T_a is saved for BOTH subgroups (A=0 and A=1) for
both metrics (log-loss, Brier).  The metric screen is forced ok=True for
all (subgroup, metric) pairs so we always get T_a + 95% CI for both
groups, matching the convention in D'Amour et al. (Figure 1).

Usage:
    python -m causal_analysis.run_synthetic \\
        --output-dir ./synthetic_results \\
        --experiment all
        --classifier hgbt        # or 'logreg' or 'both'
        --n-samples 30000
        --n-seeds 20
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .cmi import CMIConfig, estimate_cmi_stage1, estimate_cmi_stage1_discrete, estimate_cmi_stage2
from .data import OneHotBlockEncoder
from .exhaustive_search import ExhaustiveSubsetSearcher
from .gated_search import GatedSearchConfig, GatedSearcher
from .greedy_search import GreedySearchResult, GreedySubsetSearcher, SearchConfig
from .logging_utils import setup_logger, timed
from .metrics import choose_threshold_f1
from .models import PosteriorConfig
from .random_utils import set_global_seed
from .validation import (
    WeightModelConfig,
    add_gap_reduction,
    build_metric_screen_table,
    evaluate_control_set,
)
from .weight_diagnostics import write_sidecar_artifacts
try:
    from .diagnostic_plots import make_all_appendix_figures
    _HAS_PLOTS = True
except ImportError:
    _HAS_PLOTS = False


# ==========================================================================
# Classifier training (R = f̂(X))
# ==========================================================================

def _fit_classifier_get_R(
        train_df: pd.DataFrame,
        eval_df_list: Sequence[pd.DataFrame],
        x_col: str,
        y_col: str,
        classifier: str,
        seed: int,
):
    """
    Fit a classifier on (X, Y) from train_df and return predicted P(Y=1|X)
    on each dataframe in eval_df_list.

    Args:
        classifier: 'hgbt' (HistGradientBoostingClassifier with 5-fold CV
                    over max_leaf_nodes ∈ {10, 25, 50}) or 'logreg'
                    (plain LogisticRegression).
    """
    X_tr = train_df[[x_col]].values
    y_tr = train_df[y_col].astype(int).values

    if classifier == "hgbt":
        # DeAmour-style hyperparameter selection: stratified 5-fold CV
        from sklearn.model_selection import GridSearchCV
        base = HistGradientBoostingClassifier(random_state=seed)
        grid = {"max_leaf_nodes": [10, 25, 50]}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        gs = GridSearchCV(base, grid, scoring="neg_log_loss", cv=cv, n_jobs=1)
        gs.fit(X_tr, y_tr)
        model = gs.best_estimator_
    elif classifier == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(X_tr, y_tr)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    preds = []
    for df in eval_df_list:
        X_eval = df[[x_col]].values
        p = model.predict_proba(X_eval)[:, 1]
        preds.append(p)
    return preds, model


# ==========================================================================
# DGPs — graph-faithful versions
# ==========================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_dgp_1(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic 1 — minimal oracle graph (one W_pre, one W_acq).

    Causal graph:  W_pre → Y → X ← W_acq;   A ↔ W_pre, A ↔ W_acq.
    NB: W_pre does NOT directly influence X — its effect is mediated by Y.

    DGP:
        U1 ~ Bernoulli(0.4)
        Q1 ~ Bernoulli(0.8)
        A  ~ Bernoulli(σ(-0.5 + 1.2·U1 - 0.8·Q1))     # A links to both W's
        Y  ~ Bernoulli(σ(-1.2 + 2.0·U1))              # Y depends on W_pre
        X  = -0.5 + 1.5·Y + 0.8·Q1 + ε,  ε~N(0, 0.5²) # X depends on Y and W_acq only
        R  = f̂(X) (filled in by classifier)

    Conditional independences (in the population, before R is observed):
        A ⊥ Y | U1
        A ⊥ X | Y, Q1
    """
    rng = np.random.default_rng(seed)

    U1 = rng.binomial(1, 0.4, n)
    Q1 = rng.binomial(1, 0.8, n)

    p_a = _sigmoid(-0.5 + 1.2 * U1 - 0.8 * Q1)
    A = rng.binomial(1, p_a)

    p_y = _sigmoid(-1.2 + 2.0 * U1)
    Y = rng.binomial(1, p_y)

    eps = rng.normal(0, 0.5, n)
    X = -0.5 + 1.5 * Y + 0.8 * Q1 + eps

    splits = rng.choice(["train", "validate", "test"], size=n, p=[0.6, 0.2, 0.2])

    return pd.DataFrame({
        "U1": U1.astype(int),
        "Q1": Q1.astype(int),
        "A": A.astype(int),
        "gt": Y.astype(int),
        "X": X.astype(float),
        "split": splits,
    })


# def _generate_dgp_full(n: int, seed: int, with_distractors: bool) -> pd.DataFrame:
#     """
#     Graph-faithful synthetic DGP for the two-stage method.
#
#     True graph:
#         W_pre -> Y -> X -> R
#         W_acq -> X -> R
#         A is associated with W_pre and W_acq.
#
#     True supports:
#         V_Y* = {U1, U2, U3, U4, U5}
#         V_R* = {Q1, Q2, Q3}
#
#     Crucial design:
#         W_pre affects Y only.
#         W_acq affects X only.
#         W_pre does NOT directly affect X.
#     """
#     rng = np.random.default_rng(seed)
#
#     # -----------------------------
#     # True W_pre variables
#     # -----------------------------
#     U1 = rng.binomial(1, 0.40, n)
#     U2 = rng.binomial(1, 0.50, n)
#     U3 = rng.binomial(1, 0.45, n)
#     U4 = rng.binomial(1, 0.35, n)
#     U5 = rng.binomial(1, 0.30, n)
#
#     # -----------------------------
#     # True W_acq variables
#     # -----------------------------
#     Q1 = rng.binomial(1, 0.45, n)
#     Q2 = rng.binomial(1, 0.40, n)
#     Q3 = rng.binomial(1, 0.35, n)
#
#     # -----------------------------
#     # A depends on both W_pre and W_acq
#     # -----------------------------
#     p_a = _sigmoid(
#         -3.6
#         + 1.2 * U1
#         + 1.1 * U2
#         + 1.0 * U3
#         + 0.9 * U4
#         + 0.8 * U5
#         + 1.4 * Q1
#         + 1.2 * Q2
#         + 1.0 * Q3
#     )
#     A = rng.binomial(1, p_a)
#
#     # -----------------------------
#     # Y depends on W_pre only
#     # -----------------------------
#     p_y = _sigmoid(
#         -3.5
#         + 2.4 * U1
#         + 2.0 * U2
#         + 1.7 * U3
#         + 1.4 * U4
#         + 1.2 * U5
#     )
#     Y = rng.binomial(1, p_y)
#
#     # -----------------------------
#     # X depends on Y and W_acq only
#     # Acquisition variables reduce class separation.
#     # -----------------------------
#     separation = 2.6 - 1.0 * Q1 - 0.8 * Q2 - 0.6 * Q3
#     eps = rng.normal(0, 0.60, n)
#
#     X = (2 * Y - 1) * separation + eps
#
#     splits = rng.choice(["train", "validate", "test"], size=n, p=[0.6, 0.2, 0.2])
#
#     out = {
#         "U1": U1.astype(int),
#         "U2": U2.astype(int),
#         "U3": U3.astype(int),
#         "U4": U4.astype(int),
#         "U5": U5.astype(int),
#         "Q1": Q1.astype(int),
#         "Q2": Q2.astype(int),
#         "Q3": Q3.astype(int),
#         "A": A.astype(int),
#         "gt": Y.astype(int),
#         "X": X.astype(float),
#         "split": splits,
#     }
#
#     if with_distractors:
#         out["D_pre1"] = rng.binomial(1, 0.50, n).astype(int)
#         out["D_pre2"] = rng.binomial(1, 0.40, n).astype(int)
#         out["D_pre3"] = rng.binomial(1, 0.35, n).astype(int)
#         out["D_pre4"] = rng.binomial(1, 0.55, n).astype(int)
#         out["D_pre5"] = rng.binomial(1, 0.45, n).astype(int)
#
#         out["D_acq1"] = rng.binomial(1, 0.50, n).astype(int)
#         out["D_acq2"] = rng.binomial(1, 0.60, n).astype(int)
#         out["D_acq3"] = rng.binomial(1, 0.40, n).astype(int)
#
#         # True post-label variable, excluded from search.
#         p_t = _sigmoid(-2.0 + 3.0 * Y)
#         out["T1"] = rng.binomial(1, p_t).astype(int)
#
#     return pd.DataFrame(out)


def _generate_dgp_full(n: int, seed: int, with_distractors: bool) -> pd.DataFrame:
    """
    Graph-faithful synthetic DGP for Graph-Constrained Two-Stage Search.

    Target observed latent-projection graph:
        W_pre -> Y -> X -> R
        W_acq -> X -> R
        Y -> T -> X
        W_pre <-> A
        W_acq <-> A

    Underlying full DAG:
        H_Uj -> Uj
        H_Uj -> A

        H_Qj -> Qj
        H_Qj -> A

        U1,...,U5 -> Y
        Y -> T1
        Y -> X
        T1 -> X
        Q1,Q2,Q3 -> X
        X -> R

    Important:
        There is NO direct A -> Y edge.
        There is NO direct A -> X edge.
        There is NO direct A -> R edge.
        There is NO direct U -> X edge except through Y / T1.

    Therefore, in population:
        A ⟂ Y | U1,...,U5
        A ⟂ R | Y, Q1,Q2,Q3

    The final explanatory control set should be:
        VY* = {U1, U2, U3, U4, U5}
        VR* = {Q1, Q2, Q3}
        V*  = VY* ∪ VR*

    This DGP is tuned so that:
        T_a(empty) is nonzero,
        T_a(V*) should be close to zero,
        distractors are independent noise variables.
    """
    rng = np.random.default_rng(seed)

    # ================================================================
    # 1. Latent variables inducing bidirected confounding with A
    # ================================================================
    # Separate latents make each true variable independently useful,
    # avoiding the case where all Q's are redundant proxies for one H_acq.

    H_u1 = rng.normal(0.0, 1.0, n)
    H_u2 = rng.normal(0.0, 1.0, n)
    H_u3 = rng.normal(0.0, 1.0, n)
    H_u4 = rng.normal(0.0, 1.0, n)
    H_u5 = rng.normal(0.0, 1.0, n)

    H_q1 = rng.normal(0.0, 1.0, n)
    H_q2 = rng.normal(0.0, 1.0, n)
    H_q3 = rng.normal(0.0, 1.0, n)

    # ================================================================
    # 2. Observed W_pre variables
    # ================================================================
    # H_uj -> Uj

    U1 = rng.binomial(1, _sigmoid(-0.35 + 1.35 * H_u1), n)
    U2 = rng.binomial(1, _sigmoid(-0.15 + 1.25 * H_u2), n)
    U3 = rng.binomial(1, _sigmoid(-0.25 + 1.20 * H_u3), n)
    U4 = rng.binomial(1, _sigmoid(-0.55 + 1.15 * H_u4), n)
    U5 = rng.binomial(1, _sigmoid(-0.75 + 1.10 * H_u5), n)

    # ================================================================
    # 3. Observed W_acq variables
    # ================================================================
    # H_qj -> Qj

    Q1 = rng.binomial(1, _sigmoid(-0.20 + 1.35 * H_q1), n)
    Q2 = rng.binomial(1, _sigmoid(-0.35 + 1.25 * H_q2), n)
    Q3 = rng.binomial(1, _sigmoid(-0.55 + 1.20 * H_q3), n)

    # ================================================================
    # 4. Subgroup A
    # ================================================================
    # A is caused by the latent variables only.
    #
    # This induces:
    #     Uj <-> A
    #     Qj <-> A
    #
    # Do NOT put Uj or Qj directly into p_a, otherwise the graph becomes
    # Uj -> A or Qj -> A instead of bidirected latent confounding.

    p_a = _sigmoid(
        -1.35
        + 0.90 * H_u1
        + 0.85 * H_u2
        + 0.80 * H_u3
        + 0.75 * H_u4
        + 0.70 * H_u5
        + 0.95 * H_q1
        + 0.90 * H_q2
        + 0.85 * H_q3
    )
    A = rng.binomial(1, p_a, n)

    # ================================================================
    # 5. Label Y
    # ================================================================
    # W_pre -> Y only.
    #
    # No Q variables.
    # No A variable.
    #
    # This makes U the correct label-side explanatory set.

    p_y = _sigmoid(
        -3.80
        + 2.60 * U1
        + 2.25 * U2
        + 2.00 * U3
        + 1.75 * U4
        + 1.60 * U5
    )
    Y = rng.binomial(1, p_y, n)

    # ================================================================
    # 6. Post-label variable T1
    # ================================================================
    # Y -> T1.
    #
    # T1 is a true post-label variable. It can affect X, but it should
    # not be searched in the graph-constrained method.
    #
    # It is useful as a bad-control trap in unconstrained experiments.

    p_t = _sigmoid(-2.20 + 3.20 * Y)
    T1 = rng.binomial(1, p_t, n)

    # ================================================================
    # 7. Measurement X
    # ================================================================
    # Y -> X
    # T1 -> X
    # Q1,Q2,Q3 -> X
    #
    # No A -> X.
    # No direct U -> X.
    #
    # Q's affect measurement quality, so they explain residual
    # subgroup-score gaps after conditioning on Y.

    separation = (
            3.25
            - 1.60 * Q1
            - 1.15 * Q2
            - 0.95 * Q3
    )

    measurement_shift = (
            0.70 * Q1
            - 0.45 * Q2
            + 0.35 * Q3
            + 0.45 * T1
    )

    eps = rng.normal(0.0, 0.50, n)

    X = (2 * Y - 1) * separation + measurement_shift + eps

    # ================================================================
    # 8. Train / validation / test splits
    # ================================================================

    splits = rng.choice(
        ["train", "validate", "test"],
        size=n,
        p=[0.6, 0.2, 0.2],
    )

    out = {
        "U1": U1.astype(int),
        "U2": U2.astype(int),
        "U3": U3.astype(int),
        "U4": U4.astype(int),
        "U5": U5.astype(int),

        "Q1": Q1.astype(int),
        "Q2": Q2.astype(int),
        "Q3": Q3.astype(int),

        # Pipeline-compatible name for A_up.
        "A": A.astype(int),

        "gt": Y.astype(int),
        "X": X.astype(float),
        "split": splits,
    }

    if with_distractors:
        # ============================================================
        # Independent W_pre distractors
        # ============================================================
        # Not causes of Y, X, R, or A.

        out["D_pre1"] = rng.binomial(1, 0.50, n).astype(int)
        out["D_pre2"] = rng.binomial(1, 0.40, n).astype(int)
        out["D_pre3"] = rng.binomial(1, 0.35, n).astype(int)
        out["D_pre4"] = rng.binomial(1, 0.55, n).astype(int)
        out["D_pre5"] = rng.binomial(1, 0.45, n).astype(int)

        # ============================================================
        # Independent W_acq distractors
        # ============================================================
        # Not causes of X, R, or A.

        out["D_acq1"] = rng.binomial(1, 0.50, n).astype(int)
        out["D_acq2"] = rng.binomial(1, 0.60, n).astype(int)
        # out["D_acq3"] = rng.binomial(1, 0.40, n).astype(int)

        # Post-label bad-control variable.
        # Included in dataframe only so unconstrained search can be tested.
        out["T1"] = T1.astype(int)

    return pd.DataFrame(out)

def generate_dgp_clean(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic 2 — clean graph, no distractors."""
    return _generate_dgp_full(n, seed, with_distractors=False)


def generate_dgp_distractors(n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic 3 — full graph with distractors and T1."""
    return _generate_dgp_full(n, seed, with_distractors=True)


# ==========================================================================
# Shared evaluation helpers
# ==========================================================================

def _make_configs(seed: int, device):
    cmi_cfg = CMIConfig(
        n_outer_folds=3,
        inner_calibration_frac=0.2,
        random_state=seed,
        posterior=PosteriorConfig(
            hidden_dims=(64, 32), dropout=0.1, batch_size=512,
            max_epochs=60, lr=1e-3, weight_decay=1e-4, patience=8,
            num_workers=0,
        ),
    )
    weight_cfg = WeightModelConfig(
        posterior=PosteriorConfig(
            hidden_dims=(64, 32), dropout=0.1, batch_size=512,
            max_epochs=60, lr=1e-3, weight_decay=1e-4, patience=8,
            num_workers=0,
        ),
        calibration_fraction=0.2,
        random_state=seed,
        n_bootstrap=1000,
        bootstrap_alpha=0.05,
        bootstrap_seed=seed + 7,
    )
    search_cfg_fn = lambda lam: SearchConfig(
        lambda_penalty=lam,
        tolerance=1e-4,
        stage1_exact_discrete=True,
        stage1_laplace_alpha=0.0,
    )
    gated_cfg_fn = lambda lam: GatedSearchConfig(
        lambda_penalty=lam,
        hidden_dims=(64, 32), dropout=0.0, batch_size=1024,
        epochs=25, lr_head=1e-3, lr_gate=5e-3, weight_decay=1e-4,
        val_fraction=0.2, random_state=seed,
    )
    return cmi_cfg, weight_cfg, search_cfg_fn, gated_cfg_fn


def _split_df(df):
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "validate"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def _force_screen_all_true(metric_screen_df: pd.DataFrame) -> pd.DataFrame:
    """Override the metric screen so every (subgroup, metric) row is flagged.

    Default behaviour of ``build_metric_screen_table`` only flags the
    subgroup whose metric is worse than overall.  For our synthetic plots
    we want T_a and weights for BOTH groups, so we set ok=True everywhere.
    """
    out = metric_screen_df.copy()
    out["ok"] = True
    return out


def _evaluate_controls(
        val_df, test_df, encoder, control_sets, a_col, y_col, prob_col,
        threshold, device, weight_cfg, metric_screen_df, logger,
        sidecar_dir=None,
):
    """Run DeAmour validation for a dict of {name: [cols]} control sets.

    Note: metric_screen_df is forced ``ok=True`` for all subgroups so that
    T_a is computed for BOTH groups (A=0 and A=1), per user request for
    the synthetic plots.  We also restrict the returned metrics to
    log-loss and Brier only.

    If ``sidecar_dir`` is provided (Path), per-(control, subgroup) sidecar
    artifacts are written there and appendix figures are generated to
    ``sidecar_dir.parent / "figures"``.
    """
    metric_screen_df = _force_screen_all_true(metric_screen_df)
    frames = []
    all_sidecar = []
    for name, cols in control_sets.items():
        with timed(logger, f"Validation: {name}"):
            vdf = evaluate_control_set(
                df_cal=val_df, df_test=test_df,
                control_name=name, control_cols=list(cols),
                encoder=encoder, a_col=a_col, y_col=y_col, prob_col=prob_col,
                threshold=threshold, device=device,
                weight_model_config=weight_cfg,
                metric_screen_df=metric_screen_df,
                use_crossfit=True,
            )
            # Harvest sidecar artifacts BEFORE concat (which drops .attrs).
            all_sidecar.extend(vdf.attrs.get("sidecar_artifacts", []))
            frames.append(vdf)
    if sidecar_dir is not None and all_sidecar:
        sidecar_dir = Path(sidecar_dir)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        write_sidecar_artifacts(all_sidecar, out_dir=str(sidecar_dir))
        logger.info("  Wrote %d sidecar artifacts to %s",
                    len(all_sidecar), sidecar_dir)
        if _HAS_PLOTS:
            try:
                fig_dir = sidecar_dir.parent / "figures"
                fig_dir.mkdir(parents=True, exist_ok=True)
                make_all_appendix_figures(str(sidecar_dir), str(fig_dir))
                logger.info("  Wrote diagnostic figures to %s", fig_dir)
            except Exception as e:
                logger.warning("  Could not generate figures: %s", e)
    result = pd.concat(frames, ignore_index=True)
    # Restrict to log-loss and Brier only (synthetic experiments don't need
    # auroc/auprc/f1/fnr/tpr).
    result = result[result["metric"].isin(["logloss", "brier"])].reset_index(drop=True)
    result = add_gap_reduction(result, baseline_control_name="empty")
    return result


def _compute_residual_cmi(
        test_df, encoder, a_col, y_col, prob_col, v_y, v_r, device, cmi_cfg, logger,
):
    df = test_df.reset_index(drop=True)
    with timed(logger, f"Residual CMI stage-1 | V_Y={list(v_y)}"):
        res_y = estimate_cmi_stage1_discrete(df, list(v_y), a_col=a_col, y_col=y_col)
    with timed(logger, f"Residual CMI stage-2 | V_R={list(v_r)}"):
        res_r = estimate_cmi_stage2(
            df, list(v_r), encoder, a_col=a_col, y_col=y_col,
            prob_col=prob_col, device=device, config=cmi_cfg,
        )
    return {
        "Ihat_A_Y_given_VY": res_y.mi,
        "Ihat_A_R_given_Y_VR": res_r.mi,
        "ce0_stage1": res_y.ce0, "ce1_stage1": res_y.ce1,
        "ce0_stage2": res_r.ce0, "ce1_stage2": res_r.ce1,
    }


def _extract_t_a_dual_group(val_result: pd.DataFrame) -> Dict:
    """Pull T_a + CI for BOTH groups (A=0 and A=1), both metrics, plus the
    weight-diagnostic columns produced by compute_weight_diagnostics.

    Diagnostic columns are emitted once per (control, group) (under the
    logloss pass), since they are independent of the metric loop.

    Returns a flat dict with keys like
        T_a_{control}_{metric}_g{0,1}
        ci_lo_{control}_{metric}_g{0,1}
        ci_hi_{control}_{metric}_g{0,1}
        ci_contains_0_{control}_{metric}_g{0,1}
        T_a_trimmed_{control}_{metric}_g{0,1}
        n_trimmed_kept_{control}_{metric}_g{0,1}
        ess_over_n_{control}_g{0,1}
        p99_weight_{control}_g{0,1}
        max_smd_before_{control}_g{0,1}
        max_smd_after_{control}_g{0,1}
        positivity_frac_0.05_{control}_g{0,1}
        positivity_frac_0.10_{control}_g{0,1}
        ece_{control}_g{0,1}
    """
    out: Dict = {}
    diagnostic_cols_per_group = (
        "ess_over_n", "p99_weight",
        "max_smd_before", "max_smd_after",
        "positivity_frac_0.05", "positivity_frac_0.10",
        "ece",
    )
    for control in val_result["control"].unique():
        for metric in ("logloss"):
            sub = val_result[
                (val_result["control"] == control)
                & (val_result["metric"] == metric)
                & (val_result["ok"])
                ]
            for _, r in sub.iterrows():
                g = int(r["group"])
                out[f"T_a_{control}_{metric}_g{g}"] = float(r["T_a"])
                out[f"ci_lo_{control}_{metric}_g{g}"] = float(r["ci_lo"])
                out[f"ci_hi_{control}_{metric}_g{g}"] = float(r["ci_hi"])
                out[f"ci_contains_0_{control}_{metric}_g{g}"] = bool(r["ci_contains_zero"])
                # Trimmed T_a is per-metric.
                if "T_a_trimmed" in r.index:
                    v = r["T_a_trimmed"]
                    out[f"T_a_trimmed_{control}_{metric}_g{g}"] = (
                        float(v) if pd.notna(v) else float("nan")
                    )
                if "n_trimmed_kept" in r.index:
                    out[f"n_trimmed_kept_{control}_{metric}_g{g}"] = int(r["n_trimmed_kept"])
                # Weight diagnostics are constant across metrics; emit once.
                if metric == "logloss":
                    for col in diagnostic_cols_per_group:
                        if col in r.index:
                            v = r[col]
                            out[f"{col}_{control}_g{g}"] = (
                                float(v) if pd.notna(v) else float("nan")
                            )
    return out


def _print_seed_summary_table(rows: List[Dict], seed: int, classifier: str, logger) -> None:
    """Print a compact per-seed table comparing T_a(empty) vs T_a for each
    search method (logloss only).

    Defensive against missing keys: if T_a_{control}_logloss_g{0,1} is absent
    from a row (e.g. because the empty-control evaluation produced no rows),
    we print ``nan`` and continue rather than crashing.
    """
    if not rows:
        return
    methods = [r["method"] for r in rows]
    by_method = {r["method"]: r for r in rows}
    NA = float("nan")

    def _g(d, key):
        v = d.get(key, NA)
        return float(v) if v is not None else NA

    lines = []
    lines.append("=" * 78)
    lines.append(f"  SEED {seed} SUMMARY  (classifier={classifier})")
    lines.append("=" * 78)

    # One-shot diagnostic: which T_a_* keys are actually in rows[0]?
    keys_present = sorted(k for k in rows[0].keys() if k.startswith("T_a_"))
    lines.append(f"  [debug] T_a_* keys in row[0]: {keys_present}")

    for metric in ("logloss",):
        lines.append(f"  metric: {metric}")
        lines.append(
            f"  {'control':<14} | "
            f"{'T_a (g=0)':>9}  [{'lo':>7}, {'hi':>7}]  CI∋0 | "
            f"{'T_a (g=1)':>9}  [{'lo':>7}, {'hi':>7}]  CI∋0"
        )
        lines.append("  " + "-" * 76)

        first = rows[0]
        e_g0  = _g(first, f"T_a_empty_{metric}_g0")
        e_g1  = _g(first, f"T_a_empty_{metric}_g1")
        e_lo0 = _g(first, f"ci_lo_empty_{metric}_g0")
        e_hi0 = _g(first, f"ci_hi_empty_{metric}_g0")
        e_lo1 = _g(first, f"ci_lo_empty_{metric}_g1")
        e_hi1 = _g(first, f"ci_hi_empty_{metric}_g1")
        e_z0 = "Y" if first.get(f"ci_contains_0_empty_{metric}_g0", True) else "N"
        e_z1 = "Y" if first.get(f"ci_contains_0_empty_{metric}_g1", True) else "N"
        lines.append(
            f"  {'T_a(empty)':<14} | "
            f"{e_g0:>+9.4f}  [{e_lo0:>+7.4f}, {e_hi0:>+7.4f}]   {e_z0}  | "
            f"{e_g1:>+9.4f}  [{e_lo1:>+7.4f}, {e_hi1:>+7.4f}]   {e_z1}"
        )

        for m in methods:
            r = by_method[m]
            t0  = _g(r, f"T_a_searched_{metric}_g0")
            t1  = _g(r, f"T_a_searched_{metric}_g1")
            lo0 = _g(r, f"ci_lo_searched_{metric}_g0")
            hi0 = _g(r, f"ci_hi_searched_{metric}_g0")
            lo1 = _g(r, f"ci_lo_searched_{metric}_g1")
            hi1 = _g(r, f"ci_hi_searched_{metric}_g1")
            z0 = "Y" if r.get(f"ci_contains_0_searched_{metric}_g0", True) else "N"
            z1 = "Y" if r.get(f"ci_contains_0_searched_{metric}_g1", True) else "N"
            lines.append(
                f"  {m:<14} | "
                f"{t0:>+9.4f}  [{lo0:>+7.4f}, {hi0:>+7.4f}]   {z0}  | "
                f"{t1:>+9.4f}  [{lo1:>+7.4f}, {hi1:>+7.4f}]   {z1}"
            )
        lines.append("")

    msg = "\n".join(lines)
    print(msg, flush=True)
    logger.info(msg)

# ==========================================================================
# Pipeline: prepare a dataframe with R = f̂(X) predictions
# ==========================================================================

def _add_classifier_predictions(
        df: pd.DataFrame, classifier: str, seed: int, logger,
) -> pd.DataFrame:
    """Train classifier on train split, write predictions to df['prob'].

    Also prints classifier performance (log-loss, Brier, AUC, accuracy)
    on the held-out test split — overall and split by subgroup A=0 / A=1.
    """
    train_df, val_df, test_df = _split_df(df)
    (p_train, p_val, p_test), model = _fit_classifier_get_R(
        train_df, [train_df, val_df, test_df],
        x_col="X", y_col="gt", classifier=classifier, seed=seed,
    )
    df = df.copy()
    df["prob"] = np.nan
    df.loc[df["split"] == "train", "prob"] = p_train
    df.loc[df["split"] == "validate", "prob"] = p_val
    df.loc[df["split"] == "test", "prob"] = p_test

    # ---- Print + log classifier performance (overall + per-group) ----
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, accuracy_score

    def _perf(y_true, p, label):
        y_true = np.asarray(y_true).astype(int)
        p = np.asarray(p).astype(float)
        n = len(y_true)
        if n == 0:
            return None
        y_pred = (p >= 0.5).astype(int)
        try:
            ll = log_loss(y_true, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])
        except Exception:
            ll = float("nan")
        br = brier_score_loss(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
        try:
            auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        acc = accuracy_score(y_true, y_pred)
        prev = float(y_true.mean())
        return {"label": label, "n": n, "prev_Y": prev,
                "logloss": ll, "brier": br, "auc": auc, "acc": acc}

    rows = []
    rows.append(_perf(test_df["gt"].values, p_test, "overall"))
    for a in (0, 1):
        m = test_df["A"].values == a
        rows.append(_perf(test_df.loc[m, "gt"].values, p_test[m], f"A={a}"))
    rows = [r for r in rows if r is not None]

    header = (f"\n  ---- Classifier performance ({classifier}, seed={seed}) on test split ----\n"
              f"  {'group':<10} {'n':>6} {'P(Y=1)':>8} {'logloss':>9} {'brier':>8} "
              f"{'AUC':>6} {'acc':>6}")
    body_lines = []
    for r in rows:
        body_lines.append(
            f"  {r['label']:<10} {r['n']:>6d} {r['prev_Y']:>8.3f} "
            f"{r['logloss']:>9.4f} {r['brier']:>8.4f} {r['auc']:>6.3f} {r['acc']:>6.3f}"
        )
    body = "\n".join(body_lines)
    msg = header + "\n" + body
    print(msg, flush=True)
    logger.info(msg)

    return df


# ==========================================================================
# Experiment 1: Oracle graph validation
# ==========================================================================

def run_experiment_1(
        output_dir: Path, n_samples: int, seed: int, device, logger,
        classifier: str,
):
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Oracle Graph Validation  |  classifier=%s", classifier)
    logger.info("=" * 70)
    t_exp1 = time.perf_counter()

    set_global_seed(seed)
    df = generate_dgp_1(n_samples, seed=seed)
    df = _add_classifier_predictions(df, classifier, seed, logger)
    train_df, val_df, test_df = _split_df(df)
    logger.info("Generated DGP-1: n=%d (train=%d, val=%d, test=%d)",
                len(df), len(train_df), len(val_df), len(test_df))
    logger.info("Prevalence Y=1: %.3f | P(A=1): %.3f | mean prob: %.3f",
                df["gt"].mean(), df["A"].mean(), df["prob"].mean())

    a_col, y_col, prob_col = "A", "gt", "prob"
    encoder = OneHotBlockEncoder(categorical_cols=[a_col, "U1", "Q1"]).fit(df)
    threshold = choose_threshold_f1(val_df[y_col].values, val_df[prob_col].values)
    logger.info("Threshold (from val): %.4f", threshold)

    cmi_cfg, weight_cfg, _, _ = _make_configs(seed, device)
    metric_screen_df = build_metric_screen_table(
        df_test=test_df, a_col=a_col, y_col=y_col, prob_col=prob_col,
        threshold=threshold,
    )
    logger.info("Metric screen table:\n%s",
                metric_screen_df[["subgroup", "group", "metric",
                                  "metric_overall", "metric_group",
                                  "delta_overall", "ok"]].to_string(index=False))

    control_sets = {
        "empty": [],
        "U1_only": ["U1"],
        "Q1_only": ["Q1"],
        "U1_plus_Q1": ["U1", "Q1"],
    }

    logger.info("")
    logger.info("Evaluating %d control sets with DeAmour T_a (crossfit)...", len(control_sets))
    sidecar_dir = exp_dir / "diagnostics"
    validation_df = _evaluate_controls(
        val_df, test_df, encoder, control_sets, a_col, y_col, prob_col,
        threshold, device, weight_cfg, metric_screen_df, logger,
        sidecar_dir=sidecar_dir,
    )

    exp_dir = output_dir / f"experiment_1_{classifier}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("Computing residual CMI for 4 control combinations...")
    cmi_combos = [
        ([], [], "empty"),
        (["U1"], [], "VY=U1"),
        ([], ["Q1"], "VR=Q1"),
        (["U1"], ["Q1"], "VY=U1,VR=Q1"),
    ]
    cmi_rows = []
    for v_y, v_r, label in tqdm(cmi_combos, desc="Residual CMI", unit="combo"):
        cmi = _compute_residual_cmi(
            test_df, encoder, a_col, y_col, prob_col, v_y, v_r, device, cmi_cfg, logger,
        )
        cmi.update({
            "control": label,
            "V_Y": ";".join(v_y) if v_y else "(none)",
            "V_R": ";".join(v_r) if v_r else "(none)",
        })
        cmi_rows.append(cmi)
        logger.info("Residual CMI [%s]: I(A;Y|VY)=%.6f  I(A;R|Y,VR)=%.6f",
                    label, cmi["Ihat_A_Y_given_VY"], cmi["Ihat_A_R_given_Y_VR"])
    cmi_df = pd.DataFrame(cmi_rows)

    validation_df.to_csv(exp_dir / "validation_metrics.csv", index=False)
    cmi_df.to_csv(exp_dir / "residual_cmi.csv", index=False)

    logger.info("")
    logger.info("EXPERIMENT 1 — SUMMARY (classifier=%s, both groups)", classifier)
    logger.info("-" * 70)
    for control in control_sets:
        for _, row in validation_df[
            (validation_df["control"] == control) & (validation_df["ok"])
        ].iterrows():
            logger.info(
                "[%s] A=%s | metric=%s | T_a=%+.4f CI=[%+.4f,%+.4f] | CI∋0=%s | Δ=%.4f",
                control, row["group"], row["metric"], row["T_a"],
                row["ci_lo"], row["ci_hi"], row["ci_contains_zero"],
                row.get("Delta_a_m", float("nan")),
            )
    logger.info("Outputs saved to %s", exp_dir)
    logger.info("Experiment 1 total time: %.1fs", time.perf_counter() - t_exp1)
    return validation_df, cmi_df


# ==========================================================================
# Experiments 2 / 3: Search recovery
# ==========================================================================

TRUE_VY = {"U1", "U2", "U3", "U4", "U5"}
TRUE_VR = {"Q1", "Q2", "Q3"}

W_PRE_CLEAN = ["U1", "U2", "U3", "U4", "U5"]
W_ACQ_CLEAN = ["Q1", "Q2", "Q3"]

W_PRE_DISTRACTORS = W_PRE_CLEAN + ["D_pre1", "D_pre2", "D_pre3", "D_pre4", "D_pre5"]
# W_ACQ_DISTRACTORS = W_ACQ_CLEAN + ["D_acq1", "D_acq2", "D_acq3"]
W_ACQ_DISTRACTORS = W_ACQ_CLEAN + ["D_acq1", "D_acq2"]

W_UNCONSTRAINED_CLEAN = W_PRE_CLEAN + W_ACQ_CLEAN
W_UNCONSTRAINED = W_PRE_DISTRACTORS + W_ACQ_DISTRACTORS + ["T1"]
W_UNCONSTRAINED_FULL = W_PRE_DISTRACTORS + W_ACQ_DISTRACTORS + ["T1"]


def _support_metrics(recovered: set, truth: set):
    if not recovered and not truth:
        return 1.0, 1.0, 1.0
    tp = len(recovered & truth)
    prec = tp / len(recovered) if recovered else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _run_search_one_seed(
        seed: int, n_samples: int, lam: float, device, logger,
        classifier: str,
        *,
        dgp_fn,
        w_pre_pool: Sequence[str],
        w_acq_pool: Sequence[str],
        w_post_cols: Sequence[str] = (),
        methods: Sequence[str] = ("exhaustive", "greedy", "gated"),
        exp_dir: Path,
):
    set_global_seed(seed)
    df = dgp_fn(n_samples, seed=seed)
    df = _add_classifier_predictions(df, classifier, seed, logger)
    train_df, val_df, test_df = _split_df(df)

    a_col, y_col, prob_col = "A", "gt", "prob"
    # all_meta = list(w_pre_pool) + list(w_acq_pool) + list(w_post_cols) + [a_col]
    all_meta = list(dict.fromkeys(
        list(w_pre_pool) + list(w_acq_pool) + list(w_post_cols) + [a_col]
    ))
    encoder = OneHotBlockEncoder(categorical_cols=all_meta).fit(df)
    threshold = choose_threshold_f1(val_df[y_col].values, val_df[prob_col].values)
    cmi_cfg, weight_cfg, search_cfg_fn, gated_cfg_fn = _make_configs(seed, device)

    metric_screen_df = build_metric_screen_table(
        df_test=test_df, a_col=a_col, y_col=y_col, prob_col=prob_col,
        threshold=threshold,
    )

    results = []

    for method_name in tqdm(methods, desc=f"  seed {seed} methods", leave=False):
        t0 = time.perf_counter()
        logger.info("  Seed %d | classifier=%s | method=%s | lam=%.4f | starting...",
                    seed, classifier, method_name, lam)

        if method_name == "exhaustive":
            searcher = ExhaustiveSubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=a_col,
                y_col=y_col, prob_col=prob_col, device=device,
                cmi_config=cmi_cfg, search_config=search_cfg_fn(lam),
                logger=logger,
            )
            stage1 = searcher.search_stage1(list(w_pre_pool))
            searcher.search_config = search_cfg_fn(lam)
            stage2 = searcher.search_stage2(list(w_acq_pool))
        elif method_name == "greedy":
            searcher = GreedySubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=a_col,
                y_col=y_col, prob_col=prob_col, device=device,
                cmi_config=cmi_cfg, search_config=search_cfg_fn(lam),
                logger=logger,
            )
            stage1 = searcher.search_stage1(list(w_pre_pool))
            searcher.search_config = search_cfg_fn(lam)
            stage2 = searcher.search_stage2(list(w_acq_pool))
        else:  # gated
            gcfg = gated_cfg_fn(lam)
            gs = GatedSearcher(
                df_search=train_df, encoder=encoder, a_col=a_col,
                y_col=y_col, prob_col=prob_col, device=device,
                cmi_config=cmi_cfg, gated_config=gcfg, logger=logger,
            )
            stage1 = gs.search_stage1(list(w_pre_pool))
            gcfg.lambda_penalty = lam
            gs.gated_config = gcfg
            stage2 = gs.search_stage2(list(w_acq_pool))

        recovered_vy = set(stage1.selected)
        recovered_vr = set(stage2.selected)
        searched_v = list(stage1.selected) + list(stage2.selected)

        vy_prec, vy_rec, vy_f1 = _support_metrics(recovered_vy, TRUE_VY)
        vr_prec, vr_rec, vr_f1 = _support_metrics(recovered_vr, TRUE_VR)
        exact_vy = recovered_vy == TRUE_VY
        exact_vr = recovered_vr == TRUE_VR
        exact_both = exact_vy and exact_vr

        logger.info("    Stage 1: V_Y=%s | exact=%s | F1=%.2f",
                    sorted(recovered_vy), exact_vy, vy_f1)
        logger.info("    Stage 2: V_R=%s | exact=%s | F1=%.2f",
                    sorted(recovered_vr), exact_vr, vr_f1)

        cmi = _compute_residual_cmi(
            test_df, encoder, a_col, y_col, prob_col,
            stage1.selected, stage2.selected, device, cmi_cfg, logger,
        )

        # DeAmour validation: empty vs searched, BOTH groups
        control_sets = {"empty": [], "searched": searched_v}
        sidecar_dir = exp_dir / f"seed_{seed}" / method_name / "diagnostics"
        val_result = _evaluate_controls(
            val_df, test_df, encoder, control_sets, a_col, y_col, prob_col,
            threshold, device, weight_cfg, metric_screen_df, logger,
            sidecar_dir=sidecar_dir,
        )

        t_a_dict = _extract_t_a_dual_group(val_result)

        elapsed = time.perf_counter() - t0
        row = {
            "seed": seed, "classifier": classifier, "method": method_name,
            "lambda": lam, "n_train": len(train_df),
            # Stage 1
            "true_VY": ";".join(sorted(TRUE_VY)),
            "recovered_VY": ";".join(sorted(recovered_vy)),
            "exact_VY": exact_vy,
            "VY_precision": vy_prec, "VY_recall": vy_rec, "VY_F1": vy_f1,
            "J_Y": stage1.objective,
            "MI_stage1": stage1.mi if hasattr(stage1, "mi") else float("nan"),
            # Stage 2
            "true_VR": ";".join(sorted(TRUE_VR)),
            "recovered_VR": ";".join(sorted(recovered_vr)),
            "exact_VR": exact_vr,
            "VR_precision": vr_prec, "VR_recall": vr_rec, "VR_F1": vr_f1,
            "J_R": stage2.objective,
            "MI_stage2": stage2.mi if hasattr(stage2, "mi") else float("nan"),
            "exact_both": exact_both,
            # Residual CMI
            "Ihat_A_Y_given_VY": cmi["Ihat_A_Y_given_VY"],
            "Ihat_A_R_given_Y_VR": cmi["Ihat_A_R_given_Y_VR"],
            "elapsed_sec": elapsed,
        }
        row.update(t_a_dict)
        results.append(row)
        _print_seed_summary_table(results, seed, classifier, logger)

    return results


def _run_search_experiment(
        output_dir: Path,
        exp_subdir: str,
        title: str,
        n_samples: int, n_seeds: int, lam: float,
        device, logger,
        classifier: str,
        *,
        dgp_fn,
        w_pre_pool: Sequence[str],
        w_acq_pool: Sequence[str],
        w_post_cols: Sequence[str] = (),
        methods: Sequence[str] = ("exhaustive", "greedy", "gated"),
):
    logger.info("=" * 70)
    logger.info("%s  |  classifier=%s", title, classifier)
    logger.info("  n_samples=%d | n_seeds=%d | lambda=%.4f", n_samples, n_seeds, lam)
    logger.info("  TRUE V_Y* = %s", sorted(TRUE_VY))
    logger.info("  TRUE V_R* = %s", sorted(TRUE_VR))

    if list(w_pre_pool) == list(w_acq_pool):
        logger.info("  W_pre = W_acq = %s  (unconstrained)", list(w_pre_pool))
    else:
        logger.info("  W_pre candidates = %s", list(w_pre_pool))
        logger.info("  W_acq candidates = %s", list(w_acq_pool))

    logger.info("=" * 70)

    exp_dir = output_dir / f"{exp_subdir}_{classifier}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_exp = time.perf_counter()
    out_csv = exp_dir / "search_recovery_all_seeds.csv"
    for s in tqdm(range(n_seeds), desc=f"{exp_subdir} seeds", unit="seed"):
        seed = 1000 + s
        logger.info("")
        logger.info("--- Seed %d / %d (seed=%d) ---", s + 1, n_seeds, seed)
        rows = _run_search_one_seed(
            seed, n_samples, lam, device, logger, classifier,
            dgp_fn=dgp_fn,
            w_pre_pool=w_pre_pool, w_acq_pool=w_acq_pool,
            w_post_cols=w_post_cols, exp_dir=exp_dir,
            methods=methods,
        )
        # Print per-seed comparison table (raw gap vs each search method)
        # _print_seed_summary_table(rows, seed, classifier, logger)

        all_results.extend(rows)

        # Save partial CSV after each seed so progress survives crashes
        pd.DataFrame(all_results).to_csv(out_csv, index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_csv, index=False)

    agg_rows = []
    for method in methods:
        sub = results_df[results_df["method"] == method]
        agg_rows.append({
            "method": method, "classifier": classifier,
            "n_seeds": len(sub), "lambda": lam,
            "exact_VY_rate": sub["exact_VY"].mean(),
            "exact_VR_rate": sub["exact_VR"].mean(),
            "exact_both_rate": sub["exact_both"].mean(),
            "VY_precision_mean": sub["VY_precision"].mean(),
            "VY_recall_mean": sub["VY_recall"].mean(),
            "VY_F1_mean": sub["VY_F1"].mean(),
            "VR_precision_mean": sub["VR_precision"].mean(),
            "VR_recall_mean": sub["VR_recall"].mean(),
            "VR_F1_mean": sub["VR_F1"].mean(),
            "residual_CMI_stage1_mean": sub["Ihat_A_Y_given_VY"].mean(),
            "residual_CMI_stage1_std": sub["Ihat_A_Y_given_VY"].std(),
            "residual_CMI_stage2_mean": sub["Ihat_A_R_given_Y_VR"].mean(),
            "residual_CMI_stage2_std": sub["Ihat_A_R_given_Y_VR"].std(),
            "mean_elapsed_sec": sub["elapsed_sec"].mean(),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(exp_dir / "search_recovery_aggregate.csv", index=False)

    logger.info("Outputs saved to %s", exp_dir)
    logger.info("Total time: %.1fs (%.1fs per seed)",
                time.perf_counter() - t_exp,
                (time.perf_counter() - t_exp) / max(n_seeds, 1))

    # Brief aggregate logging
    logger.info("")
    logger.info("AGGREGATE SUMMARY  (classifier=%s)", classifier)
    logger.info("-" * 70)
    for method in methods:
        sub = results_df[results_df["method"] == method]
        if sub.empty:
            continue
        logger.info("[%s] (n=%d seeds)", method, len(sub))
        logger.info("  Stage 1 — exact VY recovery: %d / %d (%.0f%%)",
                    sub["exact_VY"].sum(), len(sub), 100 * sub["exact_VY"].mean())
        logger.info("  Stage 1 — VY F1: %.3f ± %.3f",
                    sub["VY_F1"].mean(), sub["VY_F1"].std())
        logger.info("  Stage 2 — exact VR recovery: %d / %d (%.0f%%)",
                    sub["exact_VR"].sum(), len(sub), 100 * sub["exact_VR"].mean())
        logger.info("  Stage 2 — VR F1: %.3f ± %.3f",
                    sub["VR_F1"].mean(), sub["VR_F1"].std())
        logger.info("  Both exact: %d / %d (%.0f%%)",
                    sub["exact_both"].sum(), len(sub), 100 * sub["exact_both"].mean())

    return results_df, agg_df


def run_experiment_2(output_dir, n_samples, n_seeds, lam, device, logger, classifier):
    """Search recovery WITHOUT distractors (clean upper-bound baseline)."""
    return _run_search_experiment(
        output_dir, exp_subdir="experiment_2_clean",
        title="EXPERIMENT 2: Search Recovery — Clean (No Distractors)",
        n_samples=n_samples, n_seeds=n_seeds, lam=lam,
        device=device, logger=logger, classifier=classifier,
        dgp_fn=generate_dgp_clean,
        w_pre_pool=W_PRE_CLEAN, w_acq_pool=W_ACQ_CLEAN, w_post_cols=[],
    )


def run_experiment_3(output_dir, n_samples, n_seeds, lam, device, logger, classifier):
    """Search recovery WITH distractors (DGP-2 / harder setting)."""
    return _run_search_experiment(
        output_dir, exp_subdir="experiment_3_distractors",
        title="EXPERIMENT 3: Search Recovery with Distractors",
        n_samples=n_samples, n_seeds=n_seeds, lam=lam,
        device=device, logger=logger, classifier=classifier,
        dgp_fn=generate_dgp_distractors,
        w_pre_pool=W_PRE_DISTRACTORS, w_acq_pool=W_ACQ_DISTRACTORS,
        w_post_cols=["T1"],
    )


def run_experiment_4_unconstrained_clean(
        output_dir, n_samples, n_seeds, lam, device, logger, classifier,
):
    """Unconstrained search on the CLEAN DGP (mirrors Experiment 2).

    Same DGP as Experiment 2 (no distractors, no T1). Stage 1 and Stage 2
    both search W_UNCONSTRAINED_CLEAN = U_1..U_5 + Q_1..Q_3. The two-stage
    objective is unchanged; only the graph partition is removed.

    Tests whether, even without nuisance variables, dropping the graph
    constraint causes either stage to select cross-stage variables (e.g.
    Stage 1 picking Q's, or Stage 2 picking U's).
    """
    return _run_search_experiment(
        output_dir, exp_subdir="experiment_4_unconstrained_clean",
        title="EXPERIMENT 4: Unconstrained Search — Clean (No Distractors)",
        n_samples=n_samples, n_seeds=n_seeds, lam=lam,
        device=device, logger=logger, classifier=classifier,
        dgp_fn=generate_dgp_clean,
        w_pre_pool=W_UNCONSTRAINED_CLEAN,
        w_acq_pool=W_UNCONSTRAINED_CLEAN,
        w_post_cols=[],
        methods=["gated", "greedy"]
    )


def run_experiment_5_unconstrained_full(
        output_dir, n_samples, n_seeds, lam, device, logger, classifier,
):
    """Unconstrained search on the DISTRACTOR DGP (mirrors Experiment 3).

    Same DGP as Experiment 3 (distractors + post-label T1). Stage 1 and
    Stage 2 both search the full union
        W_UNCONSTRAINED_FULL = U_1..U_5 + Q_1..Q_3
                              + D_pre_1..D_pre_5 + D_acq_1..D_acq_3 + T_1.

    Predicted failure mode: Stage 1 selects T_1 because conditioning on a
    descendant of Y collapses I(A;Y|.) by explaining Y away. Compare
    against Experiment 3 to isolate the contribution of the graph.
    """
    return _run_search_experiment(
        output_dir, exp_subdir="experiment_5_unconstrained_full",
        title="EXPERIMENT 5: Unconstrained Search — With Distractors and T1",
        n_samples=n_samples, n_seeds=n_seeds, lam=lam,
        device=device, logger=logger, classifier=classifier,
        dgp_fn=generate_dgp_distractors,
        w_pre_pool=W_UNCONSTRAINED_FULL,
        w_acq_pool=W_UNCONSTRAINED_FULL,
        w_post_cols=[],
        methods=["gated", "greedy"]
    )


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic experiments — graph-faithful (X-based) DGP")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["1", "2", "3", "4", "5", "all"],
                        help="Which experiment(s) to run.  '1' = oracle, '2' = clean search, "
                             "'3' = distractors search, '4' = unconstrained clean, "
                             "'5' = unconstrained with distractors, 'all' = run all five.")

    parser.add_argument("--classifier", type=str, default="both",
                        choices=["hgbt", "logreg", "both"],
                        help="Which classifier(s) to use for R = f̂(X).  "
                             "'hgbt' (HistGradientBoostingClassifier with 5-fold CV "
                             "over max_leaf_nodes ∈ {10,25,50}, matching D'Amour et al.), "
                             "'logreg' (LogisticRegression), or 'both'.")
    parser.add_argument("--n-samples", type=int, default=30000,
                        help="Total samples (split 60/20/20 into train/val/test)")
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Number of seeds for experiments 2 and 3")
    parser.add_argument("--lambda-val", type=float, default=0.003,
                        help="Lambda penalty for the search")
    parser.add_argument("--seed", type=int, default=42, help="Seed for experiment 1")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(output_dir))

    import logging as _logging
    root = _logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, _logging.FileHandler):
            root.addHandler(handler)
            break

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)
    logger.info("Classifier(s): %s", args.classifier)

    classifiers = ["hgbt", "logreg"] if args.classifier == "both" else [args.classifier]

    try:
        for clf in classifiers:
            logger.info("")
            logger.info("########################################")
            logger.info("##  classifier = %s", clf)
            logger.info("########################################")

            if args.experiment in ("1", "all"):
                run_experiment_1(output_dir, args.n_samples, args.seed, device, logger, clf)

            if args.experiment in ("2", "all"):
                run_experiment_2(output_dir, args.n_samples, args.n_seeds,
                                 args.lambda_val, device, logger, clf)

            if args.experiment in ("3", "all"):
                run_experiment_3(output_dir, args.n_samples, args.n_seeds,
                                 args.lambda_val, device, logger, clf)

            if args.experiment in ("4", "all"):
                run_experiment_4_unconstrained_clean(
                    output_dir, args.n_samples, args.n_seeds,
                    args.lambda_val, device, logger, clf,
                )

            if args.experiment in ("5", "all"):
                run_experiment_5_unconstrained_full(
                    output_dir, args.n_samples, args.n_seeds,
                    args.lambda_val, device, logger, clf,
                )

        logger.info("All synthetic experiments done.")

    except Exception:
        logger.exception("Fatal error in synthetic experiments")
        raise


if __name__ == "__main__":
    main()