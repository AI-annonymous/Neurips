"""
Synthetic experiments for the Graph-Constrained Two-Stage Search method.

Experiment 1 — Oracle graph validation (no search):
    Verify that conditioning on the TRUE causal controls drives residual CMI
    and T_a to zero on held-out data.

Experiment 2 — Search recovery with known support:
    Recover the true V_Y* and V_R* via exhaustive / greedy / gated search,
    then validate on held-out data. Repeated over 20 seeds.

Usage:
    python -m causal_analysis.run_synthetic \
        --output-dir ./synthetic_results \
        --experiment 1       # or 2 or both
        --n-samples 30000    # total (split into train/val/test)
        --n-seeds 20         # only for experiment 2
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
from tqdm import tqdm

from .cmi import CMIConfig, DiscreteCMIConfig, estimate_cmi_stage1, estimate_cmi_stage1_discrete, estimate_cmi_stage2, estimate_cmi_stage2_discrete
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


# ==========================================================================
# DGP helpers
# ==========================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_dgp_1(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic 1 — Oracle graph validation.

    Variables:
        U1 ~ Bernoulli(0.4)
        Q1 ~ Bernoulli(0.8)
        A ~ Bernoulli(σ(-0.5 + 1.2·U1 + -0.8·Q1))
        Y ~ Bernoulli(σ(-1.2 + 2.0·U1))
        R = σ(-2.0 + 3.0·Y - 1.0·Q1 + ε),  ε ~ N(0, 0.5²)

    Conditional independences:
        A ⊥ Y | U1
        A ⊥ R | Y, Q1
    """
    rng = np.random.default_rng(seed)

    U1 = rng.binomial(1, 0.4, n)
    Q1 = rng.binomial(1, 0.8, n)

    p_a = _sigmoid(-0.5 + 1.2 * U1 - 0.8 * Q1)
    A = rng.binomial(1, p_a)

    p_y = _sigmoid(-1.2 + 2.0 * U1)
    Y = rng.binomial(1, p_y)

    eps = rng.normal(0, 0.5, n)
    R = _sigmoid(-2.0 + 3.0 * Y - 1.0 * Q1 + eps)

    # Assign splits: 60% train, 20% val, 20% test
    splits = rng.choice(["train", "validate", "test"], size=n, p=[0.6, 0.2, 0.2])

    return pd.DataFrame({
        "U1": U1.astype(int),
        "Q1": Q1.astype(int),
        "A": A.astype(int),
        "gt": Y.astype(int),
        "prob": R.astype(float),
        "split": splits,
    })


def generate_dgp_2(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic 2 — Search recovery with known support (revised, strong-signal).

    True supports:
        V_Y* = {U1, U2, U3, U4, U5}
        V_R* = {Q1, Q2, Q3}

    Distractors:
        W_pre distractors: D_pre1, ..., D_pre5
        W_acq distractor:  D_acq1
        W_post (excluded): T1

    Revised DGP (relative to v1):
      - All five U's contribute meaningfully to Y (no near-zero coefficients).
      - Stronger A <- {U,Q} coupling so each true variable shows up in I(A;.).
      - Stronger Q -> R effects and lower R-noise so I(A;R|Y,V) is detectable.

        U1 ~ Bern(0.40), U2 ~ Bern(0.50), U3 ~ Bern(0.45),
        U4 ~ Bern(0.35), U5 ~ Bern(0.30)
        D_pre_j ~ Bern(0.5)
        Q1 ~ Bern(0.60), Q2 ~ Bern(0.40), Q3 ~ Bern(0.35)
        D_acq_j ~ Bern(0.5)

        A ~ Bernoulli(sigma(-2.0
                            + 1.2*U1 + 1.0*U2 + 0.9*U3 + 0.7*U4 + 0.6*U5
                            + 1.3*Q1 + 1.1*Q2 + 0.9*Q3))

        Y ~ Bernoulli(sigma(-2.0 + 2.3*U1 + 1.9*U2 + 1.5*U3 + 1.1*U4 + 0.9*U5))

        R = sigma(-2.5 + 3.0*Y - 2.0*Q1 + 1.7*Q2 + 1.4*Q3 + eps),
              eps ~ N(0, 0.25^2)
    """
    rng = np.random.default_rng(seed)

    # True W_pre variables (affect Y)
    U1 = rng.binomial(1, 0.40, n)
    U2 = rng.binomial(1, 0.50, n)
    U3 = rng.binomial(1, 0.45, n)
    U4 = rng.binomial(1, 0.35, n)
    U5 = rng.binomial(1, 0.30, n)

    # W_pre distractors (do NOT affect Y or A)
    D_pre1 = rng.binomial(1, 0.50, n)
    D_pre2 = rng.binomial(1, 0.50, n)
    D_pre3 = rng.binomial(1, 0.50, n)
    D_pre4 = rng.binomial(1, 0.50, n)
    D_pre5 = rng.binomial(1, 0.50, n)

    # True W_acq variables (affect R given Y)
    Q1 = rng.binomial(1, 0.60, n)
    Q2 = rng.binomial(1, 0.40, n)
    Q3 = rng.binomial(1, 0.35, n)

    # W_acq distractors (do NOT affect R or A)
    # Reduced from 3 -> 1 distractor: cleaner Stage-2 recovery story and
    # matches the small acquisition-side pools encountered in real medical
    # data (e.g., MIMIC W_acq = {frontal_bin, admission_location_bin}).
    D_acq1 = rng.binomial(1, 0.50, n)

    # W_post (excluded from search)
    T1 = rng.binomial(1, 0.70, n)

    # Subgroup A — depends on true U's and Q's, plus a tiny coefficient on
    # D_acq1 to make it a *near-distractor* rather than pure noise: it has
    # a real but small association with A, mimicking real-world acquisition
    # variables that are weakly correlated with subgroup membership but
    # not actually causal for the score.  The coefficient (0.1) is far
    # below the weakest true Q (Q3 at 0.9), so D_acq1's CMI drop should
    # be tiny — it will rarely be selected by a well-tuned search but
    # provides a non-trivial selection test.
    p_a = _sigmoid(
        -2.0
        + 1.2 * U1 + 1.0 * U2 + 0.9 * U3 + 0.7 * U4 + 0.6 * U5
        + 1.3 * Q1 + 1.1 * Q2 + 0.9 * Q3
        + 0.1 * D_acq1
    )
    A = rng.binomial(1, p_a)

    # Label Y — depends only on true W_pre (U1..U5); all coefficients meaningful
    p_y = _sigmoid(
        -2.0 + 2.3 * U1 + 1.9 * U2 + 1.5 * U3 + 1.1 * U4 + 0.9 * U5
    )
    Y = rng.binomial(1, p_y)

    # Score R — depends on Y and true W_acq (Q1..Q3); low noise.
    # Y coefficient is 3.0 (down from 4.0 in v0): a strong-Y term still
    # produces a sharp R(Y=0)/R(Y=1) split, but leaves enough conditional
    # variance for the Q's to be detectable in I(A;R | Y, V).  At Y=4.0,
    # mi(A;R | Y) is already ≈ noise floor (~7e-5) before any Q is added,
    # so the exhaustive search over W_acq cannot find a meaningful drop.
    eps = rng.normal(0, 0.25, n)
    R = _sigmoid(-2.5 + 3.0 * Y - 2.0 * Q1 + 1.7 * Q2 + 1.4 * Q3 + eps)

    splits = rng.choice(["train", "validate", "test"], size=n, p=[0.7, 0.15, 0.15])

    return pd.DataFrame({
        # True W_pre
        "U1": U1.astype(int), "U2": U2.astype(int), "U3": U3.astype(int),
        "U4": U4.astype(int), "U5": U5.astype(int),
        # W_pre distractors
        "D_pre1": D_pre1.astype(int), "D_pre2": D_pre2.astype(int),
        "D_pre3": D_pre3.astype(int), "D_pre4": D_pre4.astype(int),
        "D_pre5": D_pre5.astype(int),
        # True W_acq
        "Q1": Q1.astype(int), "Q2": Q2.astype(int), "Q3": Q3.astype(int),
        # W_acq distractors
        "D_acq1": D_acq1.astype(int),
        # W_post
        "T1": T1.astype(int),
        # Subgroup, label, score
        "A": A.astype(int),
        "gt": Y.astype(int),
        "prob": R.astype(float),
        "split": splits,
    })


# ==========================================================================
# Shared evaluation helpers
# ==========================================================================

def _make_configs(seed: int, device):
    cmi_cfg = CMIConfig(
        n_outer_folds=3,
        inner_calibration_frac=0.2,
        random_state=seed,
        posterior=PosteriorConfig(
            hidden_dims=(64, 32),
            dropout=0.1,
            batch_size=512,
            max_epochs=60,
            lr=1e-3,
            weight_decay=1e-4,
            patience=8,
            num_workers=0,
        ),
    )
    weight_cfg = WeightModelConfig(
        posterior=PosteriorConfig(
            hidden_dims=(64, 32),
            dropout=0.1,
            batch_size=512,
            max_epochs=60,
            lr=1e-3,
            weight_decay=1e-4,
            patience=8,
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
        # tolerance: minimum CMI improvement (in nats) required to accept a
        # greedy step or to count an exhaustive subset as better than the
        # current best.  With the strong-signal DGP-2, true Stage-1 vars
        # drop CMI by 1.7e-3 to 9e-3 (U5 is weakest at ~1.7e-3); true
        # Stage-2 vars drop CMI by 0.5e-3 to 9e-3 (Q3 is weakest at ~5e-4).
        # Distractor drops are below 1e-4.  We set 1e-4 to safely reject
        # distractors while still picking the weakest true variables.
        tolerance=1e-4,
        stage1_exact_discrete=True,
        stage1_laplace_alpha=0.5,
        stage2_exact_discrete=True,
        stage2_laplace_alpha=0.5,
        prob_col_bin="prob_bin",
        # Stage-specific beam sizes.  Stage 1 is monotonic on this DGP, so
        # default greedy (beam=1) is precise.  Stage 2 has a 3-tuple global
        # optimum {Q1,Q2,Q3} where every smaller subset INCREASES CMI
        # relative to empty (collider/non-monotonicity), so we need beam=3
        # to recover it.  Cost: with W_acq pool size 4, that's C(4,3)=4
        # extra evals at most, once.
        beam_size_stage1=1,
        beam_size_stage2=3,
    )
    gated_cfg_fn = lambda lam: GatedSearchConfig(
        # IMPORTANT: ignore caller's `lam` for synthetic gated.  At lam=0
        # the hard-concrete gates collapse to "include everything" because
        # there is no sparsity pressure.  We force a small positive penalty
        # that pushes distractor gates toward zero without dropping U5/Q3.
        lambda_penalty=max(lam, 5e-3),
        hidden_dims=(64, 32),
        dropout=0.0,
        batch_size=1024,
        # 75 epochs (was 25): the gates need long enough for the head to
        # learn which features matter before the gate distribution can
        # differentiate true from distractor.
        epochs=75,
        lr_head=1e-3,
        lr_gate=5e-3,
        weight_decay=1e-4,
        val_fraction=0.2,
        random_state=seed,
        # Sharper hard-concrete temperature (0.33 vs default 0.5) makes
        # gate probabilities more bimodal, reducing the "all gates around
        # 0.8" failure mode observed at temperature=0.5.
        hc_temperature=0.33,
    )
    return cmi_cfg, weight_cfg, search_cfg_fn, gated_cfg_fn


def _split_df(df):
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "validate"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    return train, val, test


def _evaluate_controls(
    val_df, test_df, encoder, control_sets, a_col, y_col, prob_col,
    threshold, device, weight_cfg, metric_screen_df, logger,
):
    """Run DeAmour validation for a dict of {name: [cols]} control sets."""
    frames = []
    for name, cols in control_sets.items():
        with timed(logger, f"Validation: {name}"):
            vdf = evaluate_control_set(
                df_cal=val_df, df_test=test_df,
                control_name=name, control_cols=cols,
                encoder=encoder, a_col=a_col, y_col=y_col, prob_col=prob_col,
                threshold=threshold, device=device,
                weight_model_config=weight_cfg,
                metric_screen_df=metric_screen_df,
                use_crossfit=True,  # Algorithm 2: safe here since R is from DGP, not a trained model
            )
            frames.append(vdf)
    result = pd.concat(frames, ignore_index=True)
    result = add_gap_reduction(result, baseline_control_name="empty")
    return result


def _compute_residual_cmi(
    test_df, encoder, a_col, y_col, prob_col, v_y, v_r, device, cmi_cfg, logger,
):
    """Compute held-out residual CMI for both stages (closed-form discrete)."""
    df = test_df.reset_index(drop=True)
    if "prob_bin" not in df.columns:
        # Caller forgot to binarize R; do a sensible default at 0.5.
        df = df.copy()
        df["prob_bin"] = (df[prob_col].values >= 0.5).astype(int)
    with timed(logger, f"Residual CMI stage-1 | V_Y={list(v_y)}"):
        res_y = estimate_cmi_stage1_discrete(
            df, v_y, a_col=a_col, y_col=y_col,
            config=DiscreteCMIConfig(laplace_alpha=0.5),
        )
    with timed(logger, f"Residual CMI stage-2 | V_R={list(v_r)}"):
        res_r = estimate_cmi_stage2_discrete(
            df, v_r, a_col=a_col, y_col=y_col, prob_col_bin="prob_bin",
            config=DiscreteCMIConfig(laplace_alpha=0.5),
        )
    return {
        "Ihat_A_Y_given_VY": res_y.mi,
        "Ihat_A_R_given_Y_VR": res_r.mi,
        "ce0_stage1": res_y.ce0,
        "ce1_stage1": res_y.ce1,
        "ce0_stage2": res_r.ce0,
        "ce1_stage2": res_r.ce1,
    }


# ==========================================================================
# Experiment 1: Oracle graph validation
# ==========================================================================

def run_experiment_1(output_dir: Path, n_samples: int, seed: int, device, logger):
    """
    No search. Evaluate 4 fixed control sets on the known DGP.
    Success: {U1, Q1} drives T_a ≈ 0 and residual CMI ≈ 0.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Oracle Graph Validation")
    logger.info("=" * 70)
    t_exp1 = time.perf_counter()

    set_global_seed(seed)
    df = generate_dgp_1(n_samples, seed=seed)
    train_df, val_df, test_df = _split_df(df)
    logger.info("Generated DGP-1: n=%d (train=%d, val=%d, test=%d)",
                len(df), len(train_df), len(val_df), len(test_df))
    logger.info("Prevalence Y=1: %.3f | P(A=1): %.3f",
                df["gt"].mean(), df["A"].mean())

    a_col = "A"
    y_col = "gt"
    prob_col = "prob"
    all_meta = ["U1", "Q1"]
    encoder = OneHotBlockEncoder(categorical_cols=[a_col] + all_meta).fit(df)

    threshold = choose_threshold_f1(val_df[y_col].values, val_df[prob_col].values)
    logger.info("Threshold (from val): %.4f", threshold)
    # Binarize R for the closed-form Stage-2 residual CMI; T_a continues to use
    # the continuous score.
    train_df = train_df.copy(); train_df["prob_bin"] = (train_df[prob_col].values >= threshold).astype(int)
    val_df   = val_df.copy();   val_df["prob_bin"]   = (val_df[prob_col].values   >= threshold).astype(int)
    test_df  = test_df.copy();  test_df["prob_bin"]  = (test_df[prob_col].values  >= threshold).astype(int)

    cmi_cfg, weight_cfg, _, _ = _make_configs(seed, device)

    metric_screen_df = build_metric_screen_table(
        df_test=test_df, a_col=a_col, y_col=y_col, prob_col=prob_col,
        threshold=threshold,
    )
    n_flagged = int(metric_screen_df["ok"].sum())
    logger.info("Metric screen: %d flagged (subgroup, metric) pairs out of %d",
                n_flagged, len(metric_screen_df))
    logger.info("Flagged entries:\n%s",
                metric_screen_df[metric_screen_df["ok"]][["subgroup", "metric", "delta_overall"]].to_string(index=False))

    # ---- 4 control sets ----
    control_sets = {
        "empty":                     [],
        "U1_only":              ["U1"],
        "Q1_only":              ["Q1"],
        "U1_plus_Q1":      ["U1", "Q1"],
    }

    logger.info("")
    logger.info("Evaluating %d control sets with DeAmour T_a (crossfit on test)...", len(control_sets))
    validation_df = _evaluate_controls(
        val_df, test_df, encoder, control_sets, a_col, y_col, prob_col,
        threshold, device, weight_cfg, metric_screen_df, logger,
    )

    # ---- Residual CMI for each control set combination ----
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
        cmi["control"] = label
        cmi["V_Y"] = ";".join(v_y) if v_y else "(none)"
        cmi["V_R"] = ";".join(v_r) if v_r else "(none)"
        cmi_rows.append(cmi)
        logger.info("Residual CMI [%s]: I(A;Y|VY)=%.6f  I(A;R|Y,VR)=%.6f",
                     label, cmi["Ihat_A_Y_given_VY"], cmi["Ihat_A_R_given_Y_VR"])

    cmi_df = pd.DataFrame(cmi_rows)

    # ---- Save ----
    exp_dir = output_dir / "experiment_1"
    exp_dir.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(exp_dir / "validation_metrics.csv", index=False)
    cmi_df.to_csv(exp_dir / "residual_cmi.csv", index=False)

    # ---- Print summary ----
    logger.info("")
    logger.info("EXPERIMENT 1 — SUMMARY")
    logger.info("-" * 70)
    for control in control_sets:
        sub = validation_df[
            (validation_df["control"] == control) &
            (validation_df["ok"] == True)
        ]
        if sub.empty:
            logger.info("[%s] no flagged metrics", control)
            continue
        for _, row in sub.iterrows():
            logger.info("[%s] %s=%s | metric=%s | T_a=%.4f | CI=[%.4f, %.4f] | contains_0=%s | Delta=%.4f",
                        control, a_col, row["group"], row["metric"],
                        row["T_a"], row["ci_lo"], row["ci_hi"],
                        row["ci_contains_zero"], row.get("Delta_a_m", float("nan")))

    logger.info("")
    logger.info("Residual CMI summary:")
    for _, row in cmi_df.iterrows():
        logger.info("  [%s] I(A;Y|VY)=%.6f  I(A;R|Y,VR)=%.6f",
                     row["control"], row["Ihat_A_Y_given_VY"], row["Ihat_A_R_given_Y_VR"])

    logger.info("Experiment 1 outputs saved to %s", exp_dir)
    logger.info("Experiment 1 total time: %.1fs", time.perf_counter() - t_exp1)
    return validation_df, cmi_df


# ==========================================================================
# Experiment 2: Search recovery
# ==========================================================================

TRUE_VY = {"U1", "U2", "U3", "U4", "U5"}
TRUE_VR = {"Q1", "Q2", "Q3"}

W_PRE_CANDIDATES = ["U1", "U2", "U3", "U4", "U5",
                     "D_pre1", "D_pre2", "D_pre3", "D_pre4", "D_pre5"]
W_ACQ_CANDIDATES = ["Q1", "Q2", "Q3", "D_acq1"]


def _support_metrics(recovered: set, truth: set):
    """Precision, recall, F1 for support recovery."""
    if not recovered and not truth:
        return 1.0, 1.0, 1.0
    tp = len(recovered & truth)
    prec = tp / len(recovered) if recovered else 0.0
    rec  = tp / len(truth) if truth else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _run_search_one_seed(
    seed: int, n_samples: int, lam: float, device, logger,
):
    """Run all 3 search methods for one seed. Returns a list of result dicts."""
    set_global_seed(seed)
    df = generate_dgp_2(n_samples, seed=seed)
    train_df, val_df, test_df = _split_df(df)

    a_col = "A"
    y_col = "gt"
    prob_col = "prob"
    all_meta = W_PRE_CANDIDATES + W_ACQ_CANDIDATES + ["T1", a_col]
    encoder = OneHotBlockEncoder(categorical_cols=all_meta).fit(df)

    threshold = choose_threshold_f1(val_df[y_col].values, val_df[prob_col].values)
    # Binarize R using the F1-optimal threshold from val.  Required by the
    # discrete Stage-2 CMI estimator.  T_a evaluation continues to use the
    # continuous prob column.
    train_df = train_df.copy(); train_df["prob_bin"] = (train_df[prob_col].values >= threshold).astype(int)
    val_df   = val_df.copy();   val_df["prob_bin"]   = (val_df[prob_col].values   >= threshold).astype(int)
    test_df  = test_df.copy();  test_df["prob_bin"]  = (test_df[prob_col].values  >= threshold).astype(int)
    cmi_cfg, weight_cfg, search_cfg_fn, gated_cfg_fn = _make_configs(seed, device)

    metric_screen_df = build_metric_screen_table(
        df_test=test_df, a_col=a_col, y_col=y_col, prob_col=prob_col,
        threshold=threshold,
    )

    results = []

    for method_name in tqdm(["exhaustive", "greedy", "gated"], desc=f"  seed {seed} methods", leave=False):
        t0 = time.perf_counter()
        logger.info("  Seed %d | method=%s | lam=%.4f | starting...", seed, method_name, lam)

        if method_name == "exhaustive":
            searcher = ExhaustiveSubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=a_col,
                y_col=y_col, prob_col=prob_col, device=device,
                cmi_config=cmi_cfg,
                search_config=search_cfg_fn(lam),
                logger=logger,
            )
            logger.info("    Stage 1: exhaustive over %d W_pre candidates (2^%d=%d subsets)",
                        len(W_PRE_CANDIDATES), len(W_PRE_CANDIDATES), 2**len(W_PRE_CANDIDATES))
            stage1 = searcher.search_stage1(W_PRE_CANDIDATES)
            logger.info("    Stage 1 done: V_Y=%s | mi=%.6f", stage1.selected, stage1.mi)
            searcher.search_config = search_cfg_fn(lam)
            logger.info("    Stage 2: exhaustive over %d W_acq candidates (2^%d=%d subsets)",
                        len(W_ACQ_CANDIDATES), len(W_ACQ_CANDIDATES), 2**len(W_ACQ_CANDIDATES))
            stage2 = searcher.search_stage2(W_ACQ_CANDIDATES)
            logger.info("    Stage 2 done: V_R=%s | mi=%.6f", stage2.selected, stage2.mi)

        elif method_name == "greedy":
            searcher = GreedySubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=a_col,
                y_col=y_col, prob_col=prob_col, device=device,
                cmi_config=cmi_cfg,
                search_config=search_cfg_fn(lam),
                logger=logger,
            )
            logger.info("    Stage 1: greedy over %d W_pre candidates", len(W_PRE_CANDIDATES))
            stage1 = searcher.search_stage1(W_PRE_CANDIDATES)
            logger.info("    Stage 1 done: V_Y=%s | mi=%.6f", stage1.selected, stage1.mi)
            searcher.search_config = search_cfg_fn(lam)
            logger.info("    Stage 2: greedy over %d W_acq candidates", len(W_ACQ_CANDIDATES))
            stage2 = searcher.search_stage2(W_ACQ_CANDIDATES)
            logger.info("    Stage 2 done: V_R=%s | mi=%.6f", stage2.selected, stage2.mi)

        else:  # gated
            gcfg = gated_cfg_fn(lam)
            gs = GatedSearcher(
                df_search=train_df, encoder=encoder, a_col=a_col,
                y_col=y_col, prob_col=prob_col, device=device,
                cmi_config=cmi_cfg, gated_config=gcfg, logger=logger,
            )
            logger.info("    Stage 1: gated over %d W_pre candidates", len(W_PRE_CANDIDATES))
            stage1 = gs.search_stage1(W_PRE_CANDIDATES)
            logger.info("    Stage 1 done: V_Y=%s | mi=%.6f | probs=%s",
                        stage1.selected, stage1.mi,
                        {k: f"{v:.3f}" for k, v in stage1.probabilities.items()} if hasattr(stage1, 'probabilities') and stage1.probabilities else "N/A")
            gcfg.lambda_penalty = lam
            gs.gated_config = gcfg
            logger.info("    Stage 2: gated over %d W_acq candidates", len(W_ACQ_CANDIDATES))
            stage2 = gs.search_stage2(W_ACQ_CANDIDATES)
            logger.info("    Stage 2 done: V_R=%s | mi=%.6f | probs=%s",
                        stage2.selected, stage2.mi,
                        {k: f"{v:.3f}" for k, v in stage2.probabilities.items()} if hasattr(stage2, 'probabilities') and stage2.probabilities else "N/A")

        recovered_vy = set(stage1.selected)
        recovered_vr = set(stage2.selected)
        searched_v = list(stage1.selected) + list(stage2.selected)

        # Support recovery
        vy_prec, vy_rec, vy_f1 = _support_metrics(recovered_vy, TRUE_VY)
        vr_prec, vr_rec, vr_f1 = _support_metrics(recovered_vr, TRUE_VR)
        exact_vy = recovered_vy == TRUE_VY
        exact_vr = recovered_vr == TRUE_VR
        exact_both = exact_vy and exact_vr
        logger.info("    Support: VY exact=%s (P=%.2f R=%.2f F1=%.2f) | VR exact=%s (P=%.2f R=%.2f F1=%.2f)",
                     exact_vy, vy_prec, vy_rec, vy_f1, exact_vr, vr_prec, vr_rec, vr_f1)

        # Residual CMI on test
        logger.info("    Computing held-out residual CMI...")
        cmi = _compute_residual_cmi(
            test_df, encoder, a_col, y_col, prob_col,
            stage1.selected, stage2.selected, device, cmi_cfg, logger,
        )
        logger.info("    Residual CMI: I(A;Y|VY)=%.6f | I(A;R|Y,VR)=%.6f",
                     cmi["Ihat_A_Y_given_VY"], cmi["Ihat_A_R_given_Y_VR"])

        # DeAmour validation: empty vs searched
        logger.info("    Computing DeAmour T_a (empty vs searched)...")
        control_sets = {
            "empty":    [],
            "searched": searched_v,
        }
        val_result = _evaluate_controls(
            val_df, test_df, encoder, control_sets, a_col, y_col, prob_col,
            threshold, device, weight_cfg, metric_screen_df, logger,
        )

        # Extract T_a for logloss and brier (key metrics)
        t_a_dict = {}
        for metric in ["logloss", "brier"]:
            for control in ["empty", "searched"]:
                rows = val_result[
                    (val_result["control"] == control) &
                    (val_result["metric"] == metric) &
                    (val_result["ok"] == True)
                ]
                for _, r in rows.iterrows():
                    key = f"T_a_{control}_{metric}_g{r['group']}"
                    t_a_dict[key] = r["T_a"]
                    key_ci = f"ci_contains_0_{control}_{metric}_g{r['group']}"
                    t_a_dict[key_ci] = r["ci_contains_zero"]

        elapsed = time.perf_counter() - t0
        row = {
            "seed": seed,
            "method": method_name,
            "lambda": lam,
            "n_train": len(train_df),
            # Stage 1
            "true_VY": ";".join(sorted(TRUE_VY)),
            "recovered_VY": ";".join(sorted(recovered_vy)),
            "exact_VY": exact_vy,
            "VY_precision": vy_prec,
            "VY_recall": vy_rec,
            "VY_F1": vy_f1,
            "J_Y": stage1.objective,
            "MI_stage1": stage1.mi if hasattr(stage1, "mi") else float("nan"),
            # Stage 2
            "true_VR": ";".join(sorted(TRUE_VR)),
            "recovered_VR": ";".join(sorted(recovered_vr)),
            "exact_VR": exact_vr,
            "VR_precision": vr_prec,
            "VR_recall": vr_rec,
            "VR_F1": vr_f1,
            "J_R": stage2.objective,
            "MI_stage2": stage2.mi if hasattr(stage2, "mi") else float("nan"),
            # Both
            "exact_both": exact_both,
            # Residual CMI
            "Ihat_A_Y_given_VY": cmi["Ihat_A_Y_given_VY"],
            "Ihat_A_R_given_Y_VR": cmi["Ihat_A_R_given_Y_VR"],
            # Time
            "elapsed_sec": elapsed,
        }
        row.update(t_a_dict)
        results.append(row)

        logger.info(
            "  Seed %d | %s | VY=%s (exact=%s, F1=%.2f) | VR=%s (exact=%s, F1=%.2f) | "
            "I(A;Y|VY)=%.4f | I(A;R|Y,VR)=%.4f | %.1fs",
            seed, method_name,
            sorted(recovered_vy), exact_vy, vy_f1,
            sorted(recovered_vr), exact_vr, vr_f1,
            cmi["Ihat_A_Y_given_VY"], cmi["Ihat_A_R_given_Y_VR"],
            elapsed,
        )

    return results


def run_experiment_2(
    output_dir: Path, n_samples: int, n_seeds: int, lam: float, device, logger,
):
    """
    Run search recovery over multiple seeds. Report support recovery rates
    and held-out validation metrics.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: Search Recovery with Known Support")
    logger.info("  n_samples=%d | n_seeds=%d | lambda=%.4f", n_samples, n_seeds, lam)
    logger.info("  TRUE V_Y* = %s", sorted(TRUE_VY))
    logger.info("  TRUE V_R* = %s", sorted(TRUE_VR))
    logger.info("  W_pre candidates = %s", W_PRE_CANDIDATES)
    logger.info("  W_acq candidates = %s", W_ACQ_CANDIDATES)
    logger.info("=" * 70)

    all_results = []
    t_exp2 = time.perf_counter()
    for s in tqdm(range(n_seeds), desc="Experiment 2 seeds", unit="seed"):
        seed = 1000 + s
        logger.info("")
        logger.info("--- Seed %d / %d (seed=%d) ---", s + 1, n_seeds, seed)
        rows = _run_search_one_seed(seed, n_samples, lam, device, logger)
        all_results.extend(rows)

    results_df = pd.DataFrame(all_results)

    # ---- Aggregate summary ----
    logger.info("")
    logger.info("EXPERIMENT 2 — AGGREGATE SUMMARY (n_seeds=%d, lam=%.4f)", n_seeds, lam)
    logger.info("-" * 70)

    for method in ["exhaustive", "greedy", "gated"]:
        sub = results_df[results_df["method"] == method]
        logger.info("")
        logger.info("[%s] (n=%d seeds)", method, len(sub))
        logger.info("  Stage 1 — exact VY recovery:  %d / %d (%.0f%%)",
                     sub["exact_VY"].sum(), len(sub), 100 * sub["exact_VY"].mean())
        logger.info("  Stage 1 — VY precision:       %.3f ± %.3f",
                     sub["VY_precision"].mean(), sub["VY_precision"].std())
        logger.info("  Stage 1 — VY recall:          %.3f ± %.3f",
                     sub["VY_recall"].mean(), sub["VY_recall"].std())
        logger.info("  Stage 1 — VY F1:              %.3f ± %.3f",
                     sub["VY_F1"].mean(), sub["VY_F1"].std())
        logger.info("  Stage 1 — residual I(A;Y|VY): %.4f ± %.4f",
                     sub["Ihat_A_Y_given_VY"].mean(), sub["Ihat_A_Y_given_VY"].std())
        logger.info("")
        logger.info("  Stage 2 — exact VR recovery:  %d / %d (%.0f%%)",
                     sub["exact_VR"].sum(), len(sub), 100 * sub["exact_VR"].mean())
        logger.info("  Stage 2 — VR precision:       %.3f ± %.3f",
                     sub["VR_precision"].mean(), sub["VR_precision"].std())
        logger.info("  Stage 2 — VR recall:          %.3f ± %.3f",
                     sub["VR_recall"].mean(), sub["VR_recall"].std())
        logger.info("  Stage 2 — VR F1:              %.3f ± %.3f",
                     sub["VR_F1"].mean(), sub["VR_F1"].std())
        logger.info("  Stage 2 — residual I(A;R|Y,VR): %.4f ± %.4f",
                     sub["Ihat_A_R_given_Y_VR"].mean(), sub["Ihat_A_R_given_Y_VR"].std())
        logger.info("")
        logger.info("  Both exact:                   %d / %d (%.0f%%)",
                     sub["exact_both"].sum(), len(sub), 100 * sub["exact_both"].mean())
        logger.info("  Mean time per seed:           %.1fs", sub["elapsed_sec"].mean())

    # ---- Save ----
    exp_dir = output_dir / "experiment_2"
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(exp_dir / "search_recovery_all_seeds.csv", index=False)

    # Save aggregate
    agg_rows = []
    for method in ["exhaustive", "greedy", "gated"]:
        sub = results_df[results_df["method"] == method]
        agg_rows.append({
            "method": method,
            "n_seeds": len(sub),
            "lambda": lam,
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

    logger.info("Experiment 2 outputs saved to %s", exp_dir)
    logger.info("Experiment 2 total time: %.1fs (%.1fs per seed)",
                time.perf_counter() - t_exp2,
                (time.perf_counter() - t_exp2) / max(n_seeds, 1))
    return results_df, agg_df


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic experiments for causal search method")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="Which experiment to run")
    parser.add_argument("--n-samples", type=int, default=50000,
                        help="Total samples (split 70/15/15 into train/val/test)")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds for experiment 2")
    parser.add_argument("--lambda-val", type=float, default=0.0,
                        help="Lambda penalty for experiment 2 search (use tolerance instead)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for experiment 1")
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

    try:
        if args.experiment in ("1", "both"):
            run_experiment_1(output_dir, args.n_samples, args.seed, device, logger)

        if args.experiment in ("2", "both"):
            run_experiment_2(output_dir, args.n_samples, args.n_seeds,
                             args.lambda_val, device, logger)

        logger.info("All synthetic experiments done.")

    except Exception:
        logger.exception("Fatal error in synthetic experiments")
        raise


if __name__ == "__main__":
    main()