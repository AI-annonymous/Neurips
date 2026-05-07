from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

from .cmi import CMIConfig
from .exhaustive_search import ExhaustiveSubsetSearcher
from .data import OneHotBlockEncoder, load_dataframe, split_by_column
from .feature_groups import parse_csv_list, resolve_feature_grouping
from .gated_search import GatedSearchConfig, GatedSearcher
from .greedy_search import GreedySearchResult, GreedySubsetSearcher, SearchConfig
from .logging_utils import setup_logger, timed
from .random_utils import set_global_seed
from .metrics import choose_threshold_f1
from .models import PosteriorConfig
from .validation import (
    WeightModelConfig,
    add_gap_reduction,
    build_metric_screen_table,
    empty_residual_validation_result,
    evaluate_control_set,
    stagewise_residual_validation,
)

from .weight_diagnostics import write_sidecar_artifacts
try:
    from .diagnostic_plots import make_all_appendix_figures
    _HAS_PLOTS = True
except (ImportError, SystemExit):
    _HAS_PLOTS = False


def viewposition_mapping(x):
    if pd.isnull(x):
        return 3
    x = str(x).strip().upper()

    frontal_views = {"AP", "PA", "AP AXIAL", "AP RLD", "AP LLD", "PA RLD", "PA LLD"}
    lateral_views = {"LATERAL", "XTABLE LATERAL", "LL"}
    other_views = {"LAO", "LPO", "RAO", "SWIMMERS"}

    if x in frontal_views:
        return 0
    elif x in lateral_views:
        return 1
    elif x in other_views:
        return 2
    else:
        return 2


def bin_age(x):
    if pd.isnull(x):
        return 5  # missing
    elif 0 <= x < 18:
        return 4
    elif 18 <= x < 40:
        return 3
    elif 40 <= x < 60:
        return 2
    elif 60 <= x < 80:
        return 1
    else:
        return 0


def ethnicity_mapping(x):
    if pd.isnull(x):
        return 3
    x = str(x).strip().upper()
    if x.startswith("WHITE"):
        return 0
    elif x.startswith("BLACK"):
        return 1
    elif x.startswith("ASIAN"):
        return 2
    return 3


# Human-readable labels for integer-coded subgroup values (MIMIC-CXR)
GROUP_LABELS: Dict[str, Dict[int, str]] = {
    # ---- MIMIC-CXR ----
    "age_bin": {0: "80+", 1: "60-79", 2: "40-59", 3: "18-39", 4: "0-17", 5: "missing"},
    "sex_bin": {0: "M", 1: "F", 2: "unknown"},
    "race_bin": {0: "White", 1: "Black", 2: "Asian", 3: "Other/missing"},
    "race_bin_1": {0: "White", 1: "non-White"},
    "race_bin_2": {0: "White", 1: "Black/Asian", 2: "Other/missing"},
    "sex_bin_1": {0: "M", 1: "F/unknown"},
    "insurance_bin": {0: "Medicare", 1: "Medicaid", 2: "Private", 3: "Other", 4: "No charge", 5: "missing"},
    "marital_status_bin": {0: "Married", 1: "Single", 2: "Divorced", 3: "Widowed", 4: "missing"},
    "frontal_bin": {0: "non-frontal", 1: "frontal(AP/PA)"},
    "ViewPosition_bin": {0: "frontal", 1: "lateral", 2: "other", 3: "missing"},
    "admission_type_bin": {0: "elective", 1: "urgent", 2: "emergency", 3: "observation", 4: "same-day surgical",
                           5: "missing"},
    "admission_location_bin": {0: "ER", 1: "walk-in", 2: "physician/clinic ref", 3: "transfer hospital",
                               4: "transfer SNF", 5: "psych transfer", 6: "procedure/surgical",
                               7: "info unavailable", 8: "missing"},
    "discharge_location_bin": {0: "home", 1: "home health", 2: "SNF", 3: "rehab", 4: "acute hospital",
                               5: "chronic/LTAC", 6: "healthcare facility", 7: "other facility",
                               8: "assisted living", 9: "hospice", 10: "psych", 11: "against advice",
                               12: "died", 13: "missing"},
    # ---- RSNA mammography ----
    "laterality_bin": {0: "L", 1: "R", 2: "missing"},
    "view_bin": {0: "CC", 1: "MLO", 2: "AT", 3: "other", 4: "missing"},
    "implant_bin": {0: "no implant", 1: "implant"},
    "invasive_bin": {0: "non-invasive", 1: "invasive"},
    "density_bin": {0: "A(fatty)", 1: "B(scattered)", 2: "C(heterogeneous)", 3: "D(extreme)", 4: "missing"},
    "site_id_bin": {0: "site_1", 1: "site_2"},
    "exposure_mode_bin": {0: "automatic", 1: "auto_filter", 2: "auto_time", 3: "manual", 4: "missing"},
    "photometric_bin": {0: "MONOCHROME1", 1: "MONOCHROME2", 2: "missing"},
    "pixel_intensity_bin": {0: "LOG", 1: "missing"},
    "rescale_type_bin": {0: "US", 1: "missing"},
    "voi_lut_bin": {0: "LINEAR", 1: "SIGMOID", 2: "missing"},
    # ---- VinDr mammography ----
    "density_bin": {0: "low(A/B)", 1: "high(C/D)", 2: "missing"},
    "breast_birads_bin": {0: "BI-RADS 1", 1: "BI-RADS 2", 2: "BI-RADS 3", 3: "BI-RADS 4", 4: "BI-RADS 5", 5: "missing"},
    "manufacturer_bin": {0: "Siemens", 1: "Planmed", 2: "IMS Giotto SpA", 3: "IMS srl", 4: "missing"},
    "model_name_bin": {0: "Mammomat Inspiration", 1: "Planmed Nuance", 2: "Giotto Image 3DL", 3: "Giotto Class",
                       4: "missing"},
    "photometric_bin": {0: "MONOCHROME1", 1: "MONOCHROME2", 2: "missing"},
    "presentation_lut_bin": {0: "IDENTITY", 1: "INVERSE", 2: "missing"},
    # ---- CheXpert ----
    "ap_pa_bin": {0: "AP", 1: "PA", 2: "RL", 3: "LL", 4: "missing"},
    # ---- NIH ----
    "view_pos_bin": {0: "AP", 1: "PA", 2: "missing"},
    # ---- CT-RATE ----
    "manufacturer_bin": {0: "Siemens", 1: "Philips", 2: "PNMS", 3: "missing"},
    "patient_position_bin": {0: "HFS", 1: "FFS", 2: "HFP", 3: "missing"},
}


def ethnicity_mapping_binary(x):
    """race_bin_1: 0 = White, 1 = non-White (includes NaN/other)."""
    if pd.isnull(x):
        return 1
    x = str(x).strip().upper()
    if x.startswith("WHITE"):
        return 0
    return 1


def ethnicity_mapping_ternary(x):
    """race_bin_2: 0 = White, 1 = Black/Asian, 2 = Other/missing."""
    if pd.isnull(x):
        return 2
    x = str(x).strip().upper()
    if x.startswith("WHITE"):
        return 0
    elif x.startswith("BLACK") or x.startswith("ASIAN"):
        return 1
    return 2


def _add_group_labels(df: pd.DataFrame, a_col: str) -> pd.DataFrame:
    """Add a ``group_label`` column with human-readable names."""
    label_map = GROUP_LABELS.get(a_col, {})
    if label_map:
        df["group_label"] = df["group"].apply(lambda g: label_map.get(int(g), str(g)) if pd.notna(g) else "missing")
    else:
        df["group_label"] = df["group"].astype(str)
    return df


def _jsonify_search_result(result) -> Dict:
    out = {
        "selected": list(result.selected),
        "objective": float(result.objective),
    }
    if hasattr(result, "mi"):
        out.update({"mi": float(result.mi), "ce0": float(result.ce0), "ce1": float(result.ce1)})
    if hasattr(result, "probabilities"):
        out["probabilities"] = {k: float(v) for k, v in result.probabilities.items()}
    if hasattr(result, "history"):
        hist = []
        for h in result.history:
            if isinstance(h, dict):
                hist.append(h)
            else:
                hist.append({
                    "action": h.action,
                    "subset": list(h.subset),
                    "objective": float(h.objective),
                    "mi": float(h.mi),
                    "ce0": float(h.ce0),
                    "ce1": float(h.ce1),
                })
        out["history"] = hist
    if hasattr(result, "threshold_scores"):
        out["threshold_scores"] = result.threshold_scores
    return out


def _build_control_sets(
    y_col: str,
    prob_col: str,
    fixed_metadata: Sequence[str],
    v_y: Sequence[str],
    v_r: Sequence[str],
    include_y_plus_searched: bool = True,
):
    """Build the control sets evaluated by the DeAmour T_a test.

    v_y and v_r are the two stagewise supports discovered by the search.
    searched = v_y + v_r is the combined V*.

    We report each stage's support separately and combined, both alone and
    after conditioning on Y, so the reader can read off which stage's CI is
    satisfied.  r_plus_searched is removed (score as a conditioner is not
    part of the validated CI conditions in this paper).
    """
    searched = list(v_y) + list(v_r)
    controls = {
        # "empty": [],
        # "y_only": [y_col],
        # "r_only": [prob_col],
        "fixed_metadata": list(fixed_metadata),
        # "y_plus_fixed_metadata": [y_col] + list(fixed_metadata),

        # Combined V* controls (original behaviour)
        "searched": searched,
        "y_plus_searched": [y_col] + searched,

        # Per-stage controls (new)
        "searched_vy":        list(v_y),
        "searched_vr":        list(v_r),
        # "y_plus_searched_vy": [y_col] + list(v_y),  # Stage-1 CI check
        # "y_plus_searched_vr": [y_col] + list(v_r),  # Stage-2 CI check
    }
    return controls


def run_for_subgroup(
        df: pd.DataFrame,
        args,
        subgroup_col: str,
        output_dir: Path,
        logger,
):
    train_df, val_df, test_df = split_by_column(df, split_col=args.split_col, train_name=args.train_split,
                                                val_name=args.val_split, test_name=args.test_split)
    logger.info("Split sizes: train=%d val=%d test=%d (total=%d)", len(train_df), len(val_df), len(test_df), len(df))

    grouping = resolve_feature_grouping(
        dataset_key=args.dataset_key,
        available_columns=df.columns.tolist(),
        subgroup_col=subgroup_col,
        user_w_pre=parse_csv_list(args.w_pre),
        user_w_acq=parse_csv_list(args.w_acq),
        user_w_post=parse_csv_list(args.w_post),
        include_post_in_search=args.include_post_in_search,
    )
    logger.info("A_col=%s | W_pre=%s | W_acq=%s | W_post=%s", subgroup_col, grouping.w_pre, grouping.w_acq,
                grouping.w_post)
    logger.info(
        "A_col=%s | |V_Y-candidates|=%d | |V_R-candidates|=%d | exhaustive_subsets_VY=%d | exhaustive_subsets_VR=%d | exhaustive_subsets_total=%d",
        subgroup_col,
        len(grouping.w_pre),
        len(grouping.w_acq),
        2 ** len(grouping.w_pre),
        2 ** len(grouping.w_acq),
        (2 ** len(grouping.w_pre)) + (2 ** len(grouping.w_acq)),
    )

    all_encoder_cols = sorted(set(grouping.w_pre + grouping.w_acq + grouping.w_post + [subgroup_col]))
    encoder = OneHotBlockEncoder(categorical_cols=all_encoder_cols).fit(df)

    threshold = choose_threshold_f1(val_df[args.gt_col].values, val_df[args.prob_col].values)
    logger.info("Chosen threshold on validation split: %.4f", threshold)

    metric_screen_df = build_metric_screen_table(
        df_test=test_df,
        a_col=subgroup_col,
        y_col=args.gt_col,
        prob_col=args.prob_col,
        threshold=threshold,
    )
    n_bad_metrics = int(metric_screen_df["ok"].sum())
    logger.info(
        "A_col=%s | metric-screen | n_rows=%d | n_bad_metrics=%d",
        subgroup_col,
        len(metric_screen_df),
        n_bad_metrics,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    cmi_cfg = CMIConfig(
        n_outer_folds=args.cmi_outer_folds,
        inner_calibration_frac=args.cmi_inner_cal_frac,
        random_state=args.seed,
        posterior=PosteriorConfig(
            hidden_dims=tuple(args.posterior_hidden),
            dropout=args.posterior_dropout,
            batch_size=args.posterior_batch_size,
            max_epochs=args.posterior_epochs,
            lr=args.posterior_lr,
            weight_decay=args.posterior_weight_decay,
            patience=args.posterior_patience,
            num_workers=0,
        ),
    )
    # Effective flags. If --cmi-estimator=discrete, force both stages to use
    # their closed-form contingency-table estimator; prob_col_bin must exist.
    _discrete_on = (getattr(args, "cmi_estimator", "nn") == "discrete")
    stage1_exact_discrete = bool(args.stage1_exact_discrete) or _discrete_on
    stage2_exact_discrete = _discrete_on
    prob_col_bin = f"{args.prob_col}_bin" if _discrete_on else None
    if stage2_exact_discrete and (prob_col_bin not in df.columns):
        raise ValueError(
            f"--cmi-estimator=discrete requested but '{prob_col_bin}' not found in dataframe. "
            f"Check that R binarization ran in _main_body."
        )

    search_cfg = SearchConfig(
        lambda_penalty=args.lambda_y,
        tolerance=args.greedy_tolerance,
        stage1_exact_discrete=stage1_exact_discrete,
        stage1_laplace_alpha=args.stage1_laplace_alpha,
        stage2_exact_discrete=stage2_exact_discrete,
        stage2_laplace_alpha=args.stage2_laplace_alpha,
        prob_col_bin=prob_col_bin,
    )
    gated_cfg = GatedSearchConfig(
        lambda_penalty=args.lambda_y,
        hidden_dims=tuple(args.gated_hidden),
        dropout=args.gated_dropout,
        batch_size=args.gated_batch_size,
        epochs=args.gated_epochs,
        lr_head=args.gated_lr_head,
        lr_gate=args.gated_lr_gate,
        weight_decay=args.gated_weight_decay,
        val_fraction=args.gated_val_fraction,
        random_state=args.seed,
        hc_temperature=args.hc_temperature,
        hc_gamma=args.hc_gamma,
        hc_zeta=args.hc_zeta,
    )
    weight_cfg = WeightModelConfig(
        posterior=PosteriorConfig(
            hidden_dims=tuple(args.weight_hidden),
            dropout=args.weight_dropout,
            batch_size=args.weight_batch_size,
            max_epochs=args.weight_epochs,
            lr=args.weight_lr,
            weight_decay=args.weight_weight_decay,
            patience=args.weight_patience,
            num_workers=0,
        ),
        calibration_fraction=args.weight_calibration_frac,
        random_state=args.seed,
        n_bootstrap=args.validation_bootstrap_samples,
        bootstrap_alpha=args.validation_bootstrap_alpha,
        bootstrap_seed=args.validation_bootstrap_seed,
        n_crossfit_folds=args.n_crossfit_folds,
    )

    def _expand_methods(key):
        if key == "both":
            return ["greedy", "gated"]
        elif key == "all":
            return ["greedy", "gated", "exhaustive"]
        else:
            return [key]

    s1_key = args.search_method_stage1 or args.search_method
    s2_key = args.search_method_stage2 or args.search_method
    stage1_methods = _expand_methods(s1_key)
    stage2_methods = _expand_methods(s2_key)

    logger.info("Stage-1 methods: %s | Stage-2 methods: %s", stage1_methods, stage2_methods)

    all_validation_rows: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []

    # Cache stage1 results to avoid recomputation when stage1 has fewer methods than stage2
    stage1_cache: Dict[str, GreedySearchResult] = {}

    def _run_stage1(method):
        if method in stage1_cache:
            cached = stage1_cache[method]
            logger.info("Stage-1 [%s]: using cached result => V_Y=%s (obj=%.6f, mi=%.6f)",
                        method, cached.selected, cached.objective, cached.mi)
            return cached
        n_candidates = len(grouping.w_pre)
        logger.info("=" * 70)
        logger.info("STAGE 1 | method=%s | A=%s | candidates=%s (%d vars)",
                    method, subgroup_col, grouping.w_pre, n_candidates)
        if method == "exhaustive":
            logger.info("STAGE 1 | exhaustive mode: enumerating all 2^%d = %d subsets", n_candidates, 2 ** n_candidates)
        elif method == "greedy":
            logger.info("STAGE 1 | greedy mode: forward-backward-swap search")
        else:
            logger.info("STAGE 1 | gated mode: hard-concrete differentiable search")
        logger.info("STAGE 1 | lambda_y=%.6f | n_train=%d", args.lambda_y, len(train_df))
        logger.info("=" * 70)

        effective_stage1_exact = stage1_exact_discrete or (method == "exhaustive")
        if method == "greedy":
            s = GreedySubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=subgroup_col,
                y_col=args.gt_col, prob_col=args.prob_col, device=device,
                cmi_config=cmi_cfg,
                search_config=SearchConfig(
                    lambda_penalty=args.lambda_y, tolerance=args.greedy_tolerance,
                    stage1_exact_discrete=effective_stage1_exact,
                    stage1_laplace_alpha=args.stage1_laplace_alpha,
                    stage2_exact_discrete=stage2_exact_discrete,
                    stage2_laplace_alpha=args.stage2_laplace_alpha,
                    prob_col_bin=prob_col_bin,
                ), logger=logger,
            )
            with timed(logger, f"Greedy stage-1 search for {subgroup_col}"):
                result = s.search_stage1(grouping.w_pre)
        elif method == "gated":
            gated_cfg.lambda_penalty = args.lambda_y
            s = GatedSearcher(
                df_search=train_df, encoder=encoder, a_col=subgroup_col,
                y_col=args.gt_col, prob_col=args.prob_col, device=device,
                cmi_config=cmi_cfg, gated_config=gated_cfg, logger=logger,
            )
            with timed(logger, f"Gated stage-1 search for {subgroup_col}"):
                result = s.search_stage1(grouping.w_pre)
        else:
            s = ExhaustiveSubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=subgroup_col,
                y_col=args.gt_col, prob_col=args.prob_col, device=device,
                cmi_config=cmi_cfg,
                search_config=SearchConfig(
                    lambda_penalty=args.lambda_y, tolerance=args.greedy_tolerance,
                    stage1_exact_discrete=True,
                    stage1_laplace_alpha=args.stage1_laplace_alpha,
                    stage2_exact_discrete=stage2_exact_discrete,
                    stage2_laplace_alpha=args.stage2_laplace_alpha,
                    prob_col_bin=prob_col_bin,
                ), logger=logger,
            )
            with timed(logger, f"Exhaustive stage-1 search for {subgroup_col}"):
                result = s.search_stage1(grouping.w_pre)

        logger.info("STAGE 1 result | V_Y=%s | obj=%.6f | mi=%.6f | ce0=%.6f | ce1=%.6f",
                    result.selected, result.objective, result.mi, result.ce0, result.ce1)
        if hasattr(result, 'probabilities') and result.probabilities:
            logger.info("STAGE 1 gate probabilities: %s",
                        {k: f"{v:.4f}" for k, v in result.probabilities.items()})
        stage1_cache[method] = result
        return result

    def _run_stage2(method):
        n_candidates = len(grouping.w_acq)
        logger.info("-" * 70)
        logger.info("STAGE 2 | method=%s | A=%s | candidates=%s (%d vars)",
                    method, subgroup_col, grouping.w_acq, n_candidates)
        if method == "exhaustive":
            logger.info("STAGE 2 | exhaustive mode: enumerating all 2^%d = %d subsets", n_candidates, 2 ** n_candidates)
        elif method == "greedy":
            logger.info("STAGE 2 | greedy mode: forward-backward-swap search")
        else:
            logger.info("STAGE 2 | gated mode: hard-concrete differentiable search")
        logger.info("STAGE 2 | lambda_r=%.6f | n_train=%d", args.lambda_r, len(train_df))
        logger.info("-" * 70)

        effective_stage1_exact = stage1_exact_discrete or (method == "exhaustive")
        if method == "greedy":
            s = GreedySubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=subgroup_col,
                y_col=args.gt_col, prob_col=args.prob_col, device=device,
                cmi_config=cmi_cfg,
                search_config=SearchConfig(
                    lambda_penalty=args.lambda_r, tolerance=args.greedy_tolerance,
                    stage1_exact_discrete=effective_stage1_exact,
                    stage1_laplace_alpha=args.stage1_laplace_alpha,
                    stage2_exact_discrete=stage2_exact_discrete,
                    stage2_laplace_alpha=args.stage2_laplace_alpha,
                    prob_col_bin=prob_col_bin,
                ), logger=logger,
            )
            with timed(logger, f"Greedy stage-2 search for {subgroup_col}"):
                result = s.search_stage2(grouping.w_acq)
        elif method == "gated":
            gated_cfg.lambda_penalty = args.lambda_r
            s = GatedSearcher(
                df_search=train_df, encoder=encoder, a_col=subgroup_col,
                y_col=args.gt_col, prob_col=args.prob_col, device=device,
                cmi_config=cmi_cfg, gated_config=gated_cfg, logger=logger,
            )
            with timed(logger, f"Gated stage-2 search for {subgroup_col}"):
                result = s.search_stage2(grouping.w_acq)
        else:
            s = ExhaustiveSubsetSearcher(
                df_search=train_df, encoder=encoder, a_col=subgroup_col,
                y_col=args.gt_col, prob_col=args.prob_col, device=device,
                cmi_config=cmi_cfg,
                search_config=SearchConfig(
                    lambda_penalty=args.lambda_r, tolerance=args.greedy_tolerance,
                    stage1_exact_discrete=True,
                    stage1_laplace_alpha=args.stage1_laplace_alpha,
                    stage2_exact_discrete=stage2_exact_discrete,
                    stage2_laplace_alpha=args.stage2_laplace_alpha,
                    prob_col_bin=prob_col_bin,
                ), logger=logger,
            )
            with timed(logger, f"Exhaustive stage-2 search for {subgroup_col}"):
                result = s.search_stage2(grouping.w_acq)

        logger.info("STAGE 2 result | V_R=%s | obj=%.6f | mi=%.6f | ce0=%.6f | ce1=%.6f",
                    result.selected, result.objective, result.mi, result.ce0, result.ce1)
        if hasattr(result, 'probabilities') and result.probabilities:
            logger.info("STAGE 2 gate probabilities: %s",
                        {k: f"{v:.4f}" for k, v in result.probabilities.items()})
        return result

    for s1_method in stage1_methods:
        stage1 = _run_stage1(s1_method)
        for s2_method in stage2_methods:
            method = f"{s1_method}+{s2_method}" if s1_method != s2_method else s1_method
            logger.info("*" * 70)
            logger.info("COMBINATION | stage1=%s stage2=%s => label='%s'", s1_method, s2_method, method)
            logger.info("*" * 70)
            stage2 = _run_stage2(s2_method)

            searched_v = list(stage1.selected) + list(stage2.selected)
            fixed_metadata = list(grouping.w_pre) + list(grouping.w_acq)
            control_sets = _build_control_sets(
                args.gt_col, args.prob_col, fixed_metadata,
                v_y=list(stage1.selected), v_r=list(stage2.selected),
                include_y_plus_searched=args.include_y_plus_searched,
            )
            logger.info(
                "V_Y=%s | V_R=%s | V_star=%s | control_sets=%s",
                stage1.selected, stage2.selected, searched_v, list(control_sets.keys()),
            )

            if bool(metric_screen_df["ok"].any()):
                s1_is_exhaustive = (s1_method == "exhaustive")
                residual = stagewise_residual_validation(
                    df_cal=val_df,
                    df_test=test_df,
                    encoder=encoder,
                    a_col=subgroup_col,
                    y_col=args.gt_col,
                    prob_col=args.prob_col,
                    v_y=stage1.selected,
                    v_r=stage2.selected,
                    device=device,
                    cmi_config=cmi_cfg,
                    stage1_exact_discrete=(s1_is_exhaustive or stage1_exact_discrete),
                    stage1_laplace_alpha=args.stage1_laplace_alpha,
                    stage2_exact_discrete=stage2_exact_discrete,
                    stage2_laplace_alpha=args.stage2_laplace_alpha,
                    prob_col_bin=prob_col_bin,
                )
                logger.info("Residual CMI | Ihat(A;Y|V_Y)=%.6f | Ihat(A;R|Y,V_R)=%.6f",
                            residual.stage1_mi, residual.stage2_mi)
            else:
                logger.info(
                    "Residual CMI skipped: no subgroup metric is worse than overall for A=%s",
                    subgroup_col,
                )
                residual = empty_residual_validation_result()

            logger.info("Starting DeAmour validation for %d control sets...", len(control_sets))
            validation_frames = []
            all_sidecar = []
            for control_name, control_cols in control_sets.items():
                with timed(logger, f"Validation {method}:{subgroup_col}:{control_name}"):
                    vdf = evaluate_control_set(
                        df_cal=val_df,
                        df_test=test_df,
                        control_name=control_name,
                        control_cols=control_cols,
                        encoder=encoder,
                        a_col=subgroup_col,
                        y_col=args.gt_col,
                        prob_col=args.prob_col,
                        threshold=threshold,
                        device=device,
                        weight_model_config=weight_cfg,
                        metric_screen_df=metric_screen_df,
                        use_crossfit=args.use_crossfit,
                    )
                    all_sidecar.extend(vdf.attrs.get("sidecar_artifacts", []))

                    vdf["search_method"] = method
                    vdf["search_method_stage1"] = s1_method
                    vdf["search_method_stage2"] = s2_method
                    vdf["V_Y"] = ";".join(stage1.selected)
                    vdf["V_R"] = ";".join(stage2.selected)
                    vdf["V_star"] = ";".join(searched_v)
                    vdf["Ihat_A_Y_given_VY"] = np.where(vdf["ok"].astype(bool), residual.stage1_mi, np.nan)
                    vdf["Ihat_A_R_given_Y_VR"] = np.where(vdf["ok"].astype(bool), residual.stage2_mi, np.nan)
                    validation_frames.append(vdf)
            validation_df = pd.concat(validation_frames, ignore_index=True)
            validation_df = add_gap_reduction(validation_df, baseline_control_name="empty")
            validation_df = _add_group_labels(validation_df, subgroup_col)
            all_validation_rows.append(validation_df)

            summary_rows.append({
                "A_col": subgroup_col,
                "search_method": method,
                "search_method_stage1": s1_method,
                "search_method_stage2": s2_method,
                "V_Y": ";".join(stage1.selected),
                "V_R": ";".join(stage2.selected),
                "V_star": ";".join(searched_v),
                "J_Y": float(stage1.objective),
                "J_R": float(stage2.objective),
                "Ihat_A_Y_given_VY": float(residual.stage1_mi) if bool(metric_screen_df["ok"].any()) else float("nan"),
                "Ihat_A_R_given_Y_VR": float(residual.stage2_mi) if bool(metric_screen_df["ok"].any()) else float(
                    "nan"),
                "threshold": float(threshold),
                "n_candidates_VY": int(len(grouping.w_pre)),
                "n_candidates_VR": int(len(grouping.w_acq)),
                "n_subsets_VY": int(2 ** len(grouping.w_pre)),
                "n_subsets_VR": int(2 ** len(grouping.w_acq)),
                "n_subsets_total": int((2 ** len(grouping.w_pre)) + (2 ** len(grouping.w_acq))),
                "n_bad_metrics": int(metric_screen_df["ok"].sum()),
            })

            method_dir = output_dir / subgroup_col / method
            method_dir.mkdir(parents=True, exist_ok=True)
            with open(method_dir / "stage1_search.json", "w") as f:
                json.dump(_jsonify_search_result(stage1), f, indent=2)
            with open(method_dir / "stage2_search.json", "w") as f:
                json.dump(_jsonify_search_result(stage2), f, indent=2)
            validation_df.to_csv(method_dir / "validation_metrics.csv", index=False)
            pd.DataFrame(summary_rows).to_json(method_dir / "summary.json", orient="records", indent=2)
            # NEW: persist sidecar artifacts and emit appendix figures
            if all_sidecar:
                sidecar_dir = method_dir / "diagnostics"
                sidecar_dir.mkdir(parents=True, exist_ok=True)
                write_sidecar_artifacts(all_sidecar, out_dir=str(sidecar_dir))
                logger.info("Saved %d sidecar artifacts to %s", len(all_sidecar), sidecar_dir)
                if _HAS_PLOTS:
                    try:
                        fig_dir = method_dir / "figures"
                        fig_dir.mkdir(parents=True, exist_ok=True)
                        make_all_appendix_figures(str(sidecar_dir), str(fig_dir))
                        logger.info("Saved diagnostic figures to %s", fig_dir)
                    except Exception as e:
                        logger.warning("Could not generate figures: %s", e)
            logger.info("Saved: %s/{stage1_search.json, stage2_search.json, validation_metrics.csv, summary.json}",
                        method_dir)

    final_validation = pd.concat(all_validation_rows, ignore_index=True)
    agg_csv = output_dir / subgroup_col / "all_validation_metrics.csv"
    agg_summary = output_dir / subgroup_col / "summary.csv"
    final_validation.to_csv(agg_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(agg_summary, index=False)
    logger.info("Saved aggregated results:")
    logger.info("  %s (%d rows)", agg_csv, len(final_validation))
    logger.info("  %s (%d rows)", agg_summary, len(summary_rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-constrained two-stage CMI search")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataframe (csv/parquet/pkl)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-key", type=str, default="mimic_cxr_chest")

    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validate")
    parser.add_argument("--test-split", type=str, default="test")

    parser.add_argument("--gt-col", type=str, required=True)
    parser.add_argument("--prob-col", type=str, required=True)
    parser.add_argument("--subgroup-cols", type=str, required=True, help="Comma-separated subgroup columns A")

    parser.add_argument("--w-pre", type=str, default=None, help="Comma-separated override for W_pre")
    parser.add_argument("--w-acq", type=str, default=None, help="Comma-separated override for W_acq")
    parser.add_argument("--w-post", type=str, default=None, help="Comma-separated override for W_post")
    parser.add_argument("--include-post-in-search", action="store_true")

    parser.add_argument("--search-method", type=str, default="both",
                        choices=["greedy", "gated", "exhaustive", "both", "all"])
    parser.add_argument("--search-method-stage1", type=str, default=None,
                        choices=["greedy", "gated", "exhaustive", "both", "all"],
                        help="Override search method for Stage 1 (W_pre). Defaults to --search-method.")
    parser.add_argument("--search-method-stage2", type=str, default=None,
                        choices=["greedy", "gated", "exhaustive", "both", "all"],
                        help="Override search method for Stage 2 (W_acq). Defaults to --search-method.")
    parser.add_argument("--include-y-plus-searched", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    # CMI estimator
    parser.add_argument("--cmi-outer-folds", type=int, default=3)
    parser.add_argument("--cmi-inner-cal-frac", type=float, default=0.2)
    parser.add_argument("--posterior-hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--posterior-dropout", type=float, default=0.1)
    parser.add_argument("--posterior-batch-size", type=int, default=1024)
    parser.add_argument("--posterior-epochs", type=int, default=60)
    parser.add_argument("--posterior-lr", type=float, default=1e-3)
    parser.add_argument("--posterior-weight-decay", type=float, default=1e-4)
    parser.add_argument("--posterior-patience", type=int, default=8)

    # penalties
    parser.add_argument("--lambda-y", type=float, default=0.0)
    parser.add_argument("--lambda-r", type=float, default=0.0)

    parser.add_argument("--stage1-exact-discrete", action="store_true",
                        help="Use exact discrete plugin CMI for stage 1 when supported (recommended for small discrete MIMIC debug runs)")
    parser.add_argument("--stage1-laplace-alpha", type=float, default=0.0)

    # Global CMI estimator choice. Keeps legacy default = NN.  When set to
    # 'discrete', forces both stages to use closed-form contingency-table
    # estimators (requires R to be binarized; threshold learned on val set).
    parser.add_argument(
        "--cmi-estimator", type=str, default="nn", choices=["nn", "discrete"],
        help="Global CMI estimator: 'nn' (variational NN, default) or 'discrete' "
             "(closed-form contingency tables; binarizes R using F1-optimal threshold "
             "learned on val set).",
    )
    parser.add_argument("--stage2-laplace-alpha", type=float, default=0.0,
                        help="Laplace smoothing for Stage 2 discrete CMI (only used if --cmi-estimator=discrete).")

    # greedy
    parser.add_argument("--greedy-tolerance", type=float, default=1e-4)

    # gated
    parser.add_argument("--gated-hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--gated-dropout", type=float, default=0.0)
    parser.add_argument("--gated-batch-size", type=int, default=2048)
    parser.add_argument("--gated-epochs", type=int, default=25)
    parser.add_argument("--gated-lr-head", type=float, default=1e-3)
    parser.add_argument("--gated-lr-gate", type=float, default=5e-3)
    parser.add_argument("--gated-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gated-val-fraction", type=float, default=0.2)
    parser.add_argument("--hc-temperature", type=float, default=0.5)
    parser.add_argument("--hc-gamma", type=float, default=-0.1)
    parser.add_argument("--hc-zeta", type=float, default=1.1)

    # weight model for DeAmour validation
    parser.add_argument("--weight-hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--weight-dropout", type=float, default=0.1)
    parser.add_argument("--weight-batch-size", type=int, default=1024)
    parser.add_argument("--weight-epochs", type=int, default=60)
    parser.add_argument("--weight-lr", type=float, default=1e-3)
    parser.add_argument("--weight-weight-decay", type=float, default=1e-4)
    parser.add_argument("--weight-patience", type=int, default=8)
    parser.add_argument("--weight-calibration-frac", type=float, default=0.2)
    parser.add_argument("--validation-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--validation-bootstrap-alpha", type=float, default=0.05)
    parser.add_argument("--validation-bootstrap-seed", type=int, default=123)
    parser.add_argument("--n-crossfit-folds", type=int, default=5,
                        help="Number of folds for Algorithm 2 cross-fitted weight model")
    parser.add_argument("--use-crossfit", action="store_true",
                        help="Use Algorithm 2 cross-fitting on test set (only if no separate held-out test)")

    return parser.parse_args()


def init_csv_mimic_cxr(df, logger):
    admissions_csv = "/shared/Data/MIMICIV/Rayan_files/admissions.csv"
    patient_csv = "/shared/Data/MIMICIV/Rayan_files/patients.csv"
    admissions = pd.read_csv(admissions_csv).drop_duplicates(
        subset=['subject_id'])[["subject_id",
                                "admission_type", "admission_location", "discharge_location", "insurance",
                                "marital_status"]]
    logger.info(f"admissions_csv columns: {admissions.columns}")

    patient = pd.read_csv(patient_csv)
    logger.info(f"patient columns: {patient.columns}")
    admissions_cols = [
        "subject_id",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "marital_status",
    ]

    admissions_sub = admissions[admissions_cols].copy()

    df_merged = df.merge(admissions_sub, on="subject_id", how="left")

    sex_map = {
        "M": 0,
        "F": 1,
    }

    admission_type_map = {
        "ELECTIVE": 0,
        "URGENT": 1,
        "EW EMER.": 2,
        "DIRECT EMER.": 2,
        "OBSERVATION ADMIT": 3,
        "DIRECT OBSERVATION": 3,
        "EU OBSERVATION": 3,
        "AMBULATORY OBSERVATION": 3,
        "SURGICAL SAME DAY ADMISSION": 4,
    }
    admission_location_map = {
        "EMERGENCY ROOM": 0,
        "WALK-IN/SELF REFERRAL": 1,
        "PHYSICIAN REFERRAL": 2,
        "CLINIC REFERRAL": 2,
        "TRANSFER FROM HOSPITAL": 3,
        "TRANSFER FROM SKILLED NURSING FACILITY": 4,
        "INTERNAL TRANSFER TO OR FROM PSYCH": 5,
        "PROCEDURE SITE": 6,
        "AMBULATORY SURGERY TRANSFER": 6,
        "PACU": 6,
        "INFORMATION NOT AVAILABLE": 7,
    }
    discharge_location_map = {
        "HOME": 0,
        "HOME HEALTH CARE": 1,
        "SKILLED NURSING FACILITY": 2,
        "REHAB": 3,
        "ACUTE HOSPITAL": 4,
        "CHRONIC/LONG TERM ACUTE CARE": 5,
        "HEALTHCARE FACILITY": 6,
        "OTHER FACILITY": 7,
        "ASSISTED LIVING": 8,
        "HOSPICE": 9,
        "PSYCH FACILITY": 10,
        "AGAINST ADVICE": 11,
        "DIED": 12,
    }

    insurance_map = {
        "MEDICARE": 0,
        "MEDICAID": 1,
        "PRIVATE": 2,
        "OTHER": 3,
        "NO CHARGE": 4,
    }
    marital_status_map = {
        "MARRIED": 0,
        "SINGLE": 1,
        "DIVORCED": 2,
        "WIDOWED": 3,
    }

    df_merged["race_bin"] = df_merged["race"].apply(ethnicity_mapping).astype("Int64")
    df_merged["age_bin"] = df_merged["age"].apply(bin_age).astype("Int64")
    df_merged["sex_bin"] = (
        df_merged["sex"]
        .map(sex_map)
        .fillna(2)  # missing / unknown
        .astype("Int64")
    )

    df_merged["race_bin_1"] = df_merged["race"].apply(ethnicity_mapping_binary).astype("Int64")
    df_merged["race_bin_2"] = df_merged["race"].apply(ethnicity_mapping_ternary).astype("Int64")
    df_merged["sex_bin_1"] = (
        df_merged["sex"]
        .map({"M": 0})
        .fillna(1)  # F and any missing/unknown -> 1
        .astype("Int64")
    )

    df_merged["ViewPosition_bin"] = df_merged["ViewPosition"].apply(viewposition_mapping).astype("Int64")
    df_merged["frontal_bin"] = (
        df_merged["ViewPosition"]
        .isin(["AP", "PA"])
        .astype("Int64")
    )
    df_merged["admission_type_bin"] = (
        df_merged["admission_type"]
        .map(admission_type_map)
        .fillna(5)  # missing / uncategorized
        .astype("Int64")
    )
    df_merged["admission_location_bin"] = (
        df_merged["admission_location"]
        .map(admission_location_map)
        .fillna(8)
        .astype("Int64")
    )
    df_merged["discharge_location_bin"] = (
        df_merged["discharge_location"]
        .map(discharge_location_map)
        .fillna(13)
        .astype("Int64")
    )
    df_merged["insurance_bin"] = (
        df_merged["insurance"]
        .str.strip()
        .str.upper()
        .map(insurance_map)
        .fillna(5)
        .astype("Int64")
    )
    df_merged["marital_status_bin"] = (
        df_merged["marital_status"]
        .map(marital_status_map)
        .fillna(4)
        .astype("Int64")
    )

    logger.info(f"df_merged shape: {df_merged.shape}")
    logger.info(f"df_merged columns: {df_merged.columns}")

    return df_merged


def init_csv_rsna(df, logger):
    """Integer-encode metadata columns for the RSNA mammography dataset."""
    logger.info("init_csv_rsna | input shape: %s", df.shape)

    # ---- age_bin: 0 = 0-60, 1 = 60+ ----
    if "age" in df.columns:
        df["age_bin"] = (
            df["age"]
            .apply(lambda x: 2 if pd.isnull(x) else (0 if x < 65 else 1))
            .astype("Int64")
        )
        logger.info("age_bin distribution:\n%s", df["age_bin"].value_counts().sort_index().to_string())

    # ---- laterality_bin: already {0, 1} ----
    if "laterality_bin" in df.columns:
        df["laterality_bin"] = df["laterality_bin"].fillna(2).astype("Int64")

    # ---- view_bin ----
    if "view" in df.columns:
        view_map = {"CC": 0, "MLO": 1, "AT": 2, "LM": 3, "LMO": 3, "ML": 3}
        df["view_bin"] = df["view"].map(view_map).fillna(4).astype("Int64")
        logger.info("view_bin distribution:\n%s", df["view_bin"].value_counts().sort_index().to_string())

    # ---- implant_bin ----
    if "implant" in df.columns:
        df["implant_bin"] = df["implant"].fillna(0).astype(int).clip(0, 1).astype("Int64")

    # ---- invasive_bin ----
    if "invasive" in df.columns:
        df["invasive_bin"] = df["invasive"].fillna(0).astype(int).clip(0, 1).astype("Int64")

    # ---- site_id_bin: {1, 2} -> {0, 1} ----
    if "site_id" in df.columns:
        df["site_id_bin"] = (df["site_id"].fillna(-1).astype(int) - 1).clip(0).astype("Int64")
        logger.info("site_id_bin distribution:\n%s", df["site_id_bin"].value_counts().sort_index().to_string())

    # ---- machine_id_bin: only 10 unique values, give each its own bin ----
    if "machine_id" in df.columns:
        unique_machines = sorted(df["machine_id"].dropna().unique())
        machine_map = {m: i for i, m in enumerate(unique_machines)}
        df["machine_id_bin"] = (
            df["machine_id"]
            .map(machine_map)
            .fillna(len(machine_map))
            .astype("Int64")
        )
        logger.info("machine_id_bin mapping (%d machines): %s", len(machine_map), machine_map)

    # ---- density_bin: A=0, B=1, C=2, D=3, missing=4 ----
    if "density" in df.columns:
        density_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        df["density_bin"] = (
            df["density"]
            .str.strip().str.upper()
            .map(density_map)
            .fillna(4)
            .astype("Int64")
        )
        logger.info("density_bin distribution:\n%s", df["density_bin"].value_counts().sort_index().to_string())

    # ---- Exposure Control Mode ----
    exposure_map = {
        "AUTOMATIC": 0,
        "AUTO_FILTER": 1,
        "AUTO_TIME": 2,
        "MANUAL": 3,
    }
    if "Exposure Control Mode" in df.columns:
        df["exposure_mode_bin"] = (
            df["Exposure Control Mode"]
            .str.strip().str.upper()
            .map(exposure_map)
            .fillna(4)
            .astype("Int64")
        )

    # ---- Photometric Interpretation ----
    photo_map = {"MONOCHROME1": 0, "MONOCHROME2": 1}
    if "Photometric Interpretation" in df.columns:
        df["photometric_bin"] = (
            df["Photometric Interpretation"]
            .str.strip().str.upper()
            .map(photo_map)
            .fillna(2)
            .astype("Int64")
        )

    # ---- Pixel Intensity Relationship ----
    pixel_map = {"LOG": 0}
    if "Pixel Intensity Relationship" in df.columns:
        df["pixel_intensity_bin"] = (
            df["Pixel Intensity Relationship"]
            .str.strip().str.upper()
            .map(pixel_map)
            .fillna(1)
            .astype("Int64")
        )

    # ---- Rescale Type ----
    rescale_map = {"US": 0}
    if "Rescale Type" in df.columns:
        df["rescale_type_bin"] = (
            df["Rescale Type"]
            .str.strip().str.upper()
            .map(rescale_map)
            .fillna(1)
            .astype("Int64")
        )

    # ---- VOI LUT Function ----
    voi_map = {"LINEAR": 0, "SIGMOID": 1}
    if "VOI LUT Function" in df.columns:
        df["voi_lut_bin"] = (
            df["VOI LUT Function"]
            .str.strip().str.upper()
            .map(voi_map)
            .fillna(2)
            .astype("Int64")
        )

    logger.info("init_csv_rsna | output shape: %s", df.shape)
    logger.info("init_csv_rsna | new _bin columns: %s",
                [c for c in df.columns if c.endswith("_bin") and c not in ["Predictions_bin",
                                                                           "H1_vascular calcifications_bin",
                                                                           "H2_scattered calcifications_bin",
                                                                           "H3_benign appearing calcifications_bin"]])

    return df


def init_csv_rsna_mirai(df, logger, gt_col="gt_cancer"):
    """RSNA used as external evaluation set for MIRAI (a SOTA risk predictor).

    Runs the standard RSNA preprocessing, then overwrites the `split` column
    with a stratified 65/15/20 train/validate/test assignment (fixed seed 42).
    Stratification is on the cancer label (`gt_col`), because positives are
    very rare (~2%) and a plain random split would leave too few cases in the
    small val/test partitions.

    Pipeline:
      - 65% train  -> CMI search
      - 15% val    -> DeAmour weight model fit
      - 20% test   -> T_a evaluation + bootstrap CI
    """
    df = init_csv_rsna(df, logger)

    from sklearn.model_selection import train_test_split

    if gt_col not in df.columns:
        raise ValueError(f"init_csv_rsna_mirai: stratification column '{gt_col}' not found in df.")

    y = df[gt_col].astype(int).values
    idx = np.arange(len(df))

    # First split: 20% test (stratified)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.20, random_state=42, stratify=y,
    )
    # Second split: of remaining 80%, take 15/80 = 0.1875 as validate (stratified)
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=(0.15 / 0.80), random_state=42, stratify=y_trainval,
    )

    df = df.copy()
    df["split"] = ""
    df.loc[df.index[idx_train], "split"] = "train"
    df.loc[df.index[idx_val],   "split"] = "validate"
    df.loc[df.index[idx_test],  "split"] = "test"

    for name in ["train", "validate", "test"]:
        mask = df["split"] == name
        n = int(mask.sum())
        n_pos = int(df.loc[mask, gt_col].astype(int).sum())
        prev = n_pos / max(n, 1)
        logger.info(
            "init_csv_rsna_mirai | split=%-8s  n=%d  pos=%d  prev=%.4f",
            name, n, n_pos, prev,
        )
    return df


def init_csv_vindr(df, logger):
    """Integer-encode metadata columns for the VinDr mammography dataset."""
    logger.info("init_csv_vindr | input shape: %s", df.shape)

    # ---- laterality_bin: already {0, 1}, ensure int ----
    if "laterality_bin" in df.columns:
        df["laterality_bin"] = df["laterality_bin"].fillna(2).astype("Int64")
    elif "laterality" in df.columns:
        lat_map = {"L": 0, "R": 1}
        df["laterality_bin"] = df["laterality"].map(lat_map).fillna(2).astype("Int64")

    # ---- view_bin ----
    if "view" in df.columns:
        view_map = {"CC": 0, "MLO": 1}
        df["view_bin"] = df["view"].str.strip().str.upper().map(view_map).fillna(2).astype("Int64")
        logger.info("view_bin distribution:\n%s", df["view_bin"].value_counts().sort_index().to_string())

    # ---- density_bin: A/B -> 0 (low), C/D -> 1 (high) ----
    if "breast_density" in df.columns:
        def _density_binary(x):
            if pd.isnull(x):
                return 2
            x = str(x).strip().upper()
            if "A" in x or "B" in x:
                return 0
            elif "C" in x or "D" in x:
                return 1
            return 2

        df["density_bin"] = df["breast_density"].apply(_density_binary).astype("Int64")
        logger.info("density_bin distribution:\n%s", df["density_bin"].value_counts().sort_index().to_string())

    # ---- breast_birads_bin ----
    if "breast_birads" in df.columns:
        birads_map = {"BI-RADS 1": 0, "BI-RADS 2": 1, "BI-RADS 3": 2, "BI-RADS 4": 3, "BI-RADS 5": 4}
        df["breast_birads_bin"] = (
            df["breast_birads"]
            .str.strip().str.upper()
            .map({k.upper(): v for k, v in birads_map.items()})
            .fillna(5)
            .astype("Int64")
        )
        logger.info("breast_birads_bin distribution:\n%s",
                    df["breast_birads_bin"].value_counts().sort_index().to_string())

    # ---- Manufacturer ----
    manufacturer_map = {
        "SIEMENS": 0,
        "PLANMED": 1,
        "IMS GIOTTO S.P.A.": 2,
        "IMS S.R.L.": 3,
    }
    if "Manufacturer" in df.columns:
        df["manufacturer_bin"] = (
            df["Manufacturer"]
            .str.strip().str.upper()
            .map(manufacturer_map)
            .fillna(4)
            .astype("Int64")
        )
        logger.info("manufacturer_bin distribution:\n%s",
                    df["manufacturer_bin"].value_counts().sort_index().to_string())

    # ---- ManufacturersModelName ----
    model_map = {
        "MAMMOMAT INSPIRATION": 0,
        "PLANMED NUANCE": 1,
        "GIOTTO IMAGE 3DL": 2,
        "GIOTTO CLASS": 3,
    }
    if "ManufacturersModelName" in df.columns:
        df["model_name_bin"] = (
            df["ManufacturersModelName"]
            .str.strip().str.upper()
            .map(model_map)
            .fillna(4)
            .astype("Int64")
        )
        logger.info("model_name_bin distribution:\n%s", df["model_name_bin"].value_counts().sort_index().to_string())

    # ---- PhotometricInterpretation ----
    photo_map = {"MONOCHROME1": 0, "MONOCHROME2": 1}
    if "PhotometricInterpretation" in df.columns:
        df["photometric_bin"] = (
            df["PhotometricInterpretation"]
            .str.strip().str.upper()
            .map(photo_map)
            .fillna(2)
            .astype("Int64")
        )

    # ---- VOILUTFunction ----
    voi_map = {"LINEAR": 0, "SIGMOID": 1}
    if "VOILUTFunction" in df.columns:
        df["voi_lut_bin"] = (
            df["VOILUTFunction"]
            .str.strip().str.upper()
            .map(voi_map)
            .fillna(2)
            .astype("Int64")
        )

    # ---- PresentationLUTShape ----
    lut_shape_map = {"IDENTITY": 0, "INVERSE": 1}
    if "PresentationLUTShape" in df.columns:
        df["presentation_lut_bin"] = (
            df["PresentationLUTShape"]
            .str.strip().str.upper()
            .map(lut_shape_map)
            .fillna(2)
            .astype("Int64")
        )

    logger.info("init_csv_vindr | output shape: %s", df.shape)
    logger.info("init_csv_vindr | new _bin columns: %s",
                [c for c in df.columns if c.endswith("_bin") and c not in [
                    "Predictions_bin",
                    "H1_benign calcifications_bin", "H2_stable calcifications_bin",
                    "H3_progressive calcifications_bin", "H4_scattered calcifications_bin",
                    "H5_nodules_bin"]])

    return df


def init_csv_chexpert(df, logger):
    """Integer-encode metadata columns for the CheXpert dataset (DenseNet121)."""
    logger.info("init_csv_chexpert | input shape: %s", df.shape)

    # ---- age_bin (same bins as MIMIC) ----
    if "age" in df.columns:
        def _bin_age(x):
            if pd.isnull(x):
                return 5
            elif 0 <= x < 18:
                return 4
            elif 18 <= x < 40:
                return 3
            elif 40 <= x < 60:
                return 2
            elif 60 <= x < 80:
                return 1
            else:
                return 0

        df["age_bin"] = df["age"].apply(_bin_age).astype("Int64")
        logger.info("age_bin distribution:\n%s", df["age_bin"].value_counts().sort_index().to_string())

    # ---- sex_bin: already {0=M, 1=F} ----
    if "sex" in df.columns:
        df["sex_bin"] = df["sex"].fillna(2).astype("Int64")
        logger.info("sex_bin distribution:\n%s", df["sex_bin"].value_counts().sort_index().to_string())

    # ---- race_bin ----
    if "PRIMARY_RACE" in df.columns:
        def _cat_race(r):
            if isinstance(r, str):
                if r.startswith("White"):
                    return 0
                elif r.startswith("Black"):
                    return 1
                elif r.startswith("Asian"):
                    return 2
            return 3

        df["race_bin"] = df["PRIMARY_RACE"].apply(_cat_race).astype("Int64")
        logger.info("race_bin distribution:\n%s", df["race_bin"].value_counts().sort_index().to_string())

    # ---- frontal_bin: Frontal=1, Lateral=0 ----
    if "Frontal/Lateral" in df.columns:
        df["frontal_bin"] = (
            df["Frontal/Lateral"]
            .str.strip()
            .map({"Frontal": 1, "Lateral": 0})
            .fillna(2)
            .astype("Int64")
        )
        logger.info("frontal_bin distribution:\n%s", df["frontal_bin"].value_counts().sort_index().to_string())

    # ---- ap_pa_bin: AP=0, PA=1, RL=2, LL=3, missing=4 ----
    if "AP/PA" in df.columns:
        df["ap_pa_bin"] = (
            df["AP/PA"]
            .str.strip().str.upper()
            .map({"AP": 0, "PA": 1, "RL": 2, "LL": 3})
            .fillna(4)
            .astype("Int64")
        )
        logger.info("ap_pa_bin distribution:\n%s", df["ap_pa_bin"].value_counts().sort_index().to_string())

    logger.info("init_csv_chexpert | output shape: %s", df.shape)
    logger.info("init_csv_chexpert | _bin columns: %s",
                [c for c in df.columns if c.endswith("_bin")])

    return df


def init_csv_nih(df, logger):
    """Integer-encode metadata columns for the NIH ChestX-ray14 dataset."""
    logger.info("init_csv_nih | input shape: %s", df.shape)

    # ---- age_bin (same bins as MIMIC) ----
    if "age" in df.columns:
        def _bin_age(x):
            if pd.isnull(x):
                return 5
            elif 0 <= x < 18:
                return 4
            elif 18 <= x < 40:
                return 3
            elif 40 <= x < 60:
                return 2
            elif 60 <= x < 80:
                return 1
            else:
                return 0

        df["age_bin"] = df["age"].apply(_bin_age).astype("Int64")
        logger.info("age_bin distribution:\n%s", df["age_bin"].value_counts().sort_index().to_string())

    # ---- sex_bin: M=0, F=1 ----
    if "Patient Sex" in df.columns:
        df["sex_bin"] = (
            df["Patient Sex"]
            .str.strip().str.upper()
            .map({"M": 0, "F": 1})
            .fillna(2)
            .astype("Int64")
        )
        logger.info("sex_bin distribution:\n%s", df["sex_bin"].value_counts().sort_index().to_string())

    # ---- view_pos_bin: AP=0, PA=1 ----
    if "View Position" in df.columns:
        df["view_pos_bin"] = (
            df["View Position"]
            .str.strip().str.upper()
            .map({"AP": 0, "PA": 1})
            .fillna(2)
            .astype("Int64")
        )
        logger.info("view_pos_bin distribution:\n%s", df["view_pos_bin"].value_counts().sort_index().to_string())

    logger.info("init_csv_nih | output shape: %s", df.shape)
    logger.info("init_csv_nih | _bin columns: %s",
                [c for c in df.columns if c.endswith("_bin")])

    return df


def init_csv_ctrate(df, logger):
    """Integer-encode metadata columns for the CT-RATE lung CT dataset."""
    logger.info("init_csv_ctrate | input shape: %s", df.shape)

    # ---- Remap splits: original train -> train(85%)/validate(15%), original validation -> test ----
    rng = np.random.default_rng(42)
    train_mask = df["split"] == "train"
    train_idx = df.index[train_mask].values
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * 0.15)
    val_idx = train_idx[:n_val]
    df.loc[val_idx, "split"] = "validate"
    df.loc[df["split"] == "validation", "split"] = "test"
    logger.info("Split remapping: original train -> train(%d)/validate(%d), original validation -> test(%d)",
                int((df["split"] == "train").sum()), len(val_idx), int((df["split"] == "test").sum()))

    # ---- age_bin: parse "42Y" format, then bin ----
    if "PatientAge" in df.columns:
        def _parse_age(x):
            if pd.isnull(x):
                return np.nan
            s = str(x).strip().upper().replace("Y", "")
            try:
                return float(s)
            except ValueError:
                return np.nan

        df["age"] = df["PatientAge"].apply(_parse_age)

        def _bin_age(x):
            if pd.isnull(x):
                return 5
            elif 0 <= x < 18:
                return 4
            elif 18 <= x < 40:
                return 3
            elif 40 <= x < 60:
                return 2
            elif 60 <= x < 80:
                return 1
            else:
                return 0

        df["age_bin"] = df["age"].apply(_bin_age).astype("Int64")
        logger.info("age_bin distribution:\n%s", df["age_bin"].value_counts().sort_index().to_string())

    # ---- sex_bin: M=0, F=1 ----
    if "PatientSex" in df.columns:
        df["sex_bin"] = (
            df["PatientSex"]
            .str.strip().str.upper()
            .map({"M": 0, "F": 1})
            .fillna(2)
            .astype("Int64")
        )
        logger.info("sex_bin distribution:\n%s", df["sex_bin"].value_counts().sort_index().to_string())

    # ---- manufacturer_bin: Siemens variants -> 0, Philips -> 1, PNMS -> 2 ----
    if "Manufacturer" in df.columns:
        def _cat_manufacturer(x):
            if pd.isnull(x):
                return 3
            x = str(x).strip().upper()
            if "SIEMENS" in x or "HEALTHINEERS" in x:
                return 0
            elif "PHILIPS" in x:
                return 1
            elif "PNMS" in x:
                return 2
            return 3

        df["manufacturer_bin"] = df["Manufacturer"].apply(_cat_manufacturer).astype("Int64")
        logger.info("manufacturer_bin distribution:\n%s",
                    df["manufacturer_bin"].value_counts().sort_index().to_string())

    # ---- model_name_bin: factorize ----
    if "ManufacturerModelName" in df.columns:
        codes, uniques = pd.factorize(df["ManufacturerModelName"].fillna("missing"))
        df["model_name_bin"] = codes.astype(int)
        logger.info("model_name_bin mapping: %s", dict(enumerate(uniques)))
        logger.info("model_name_bin distribution:\n%s", df["model_name_bin"].value_counts().sort_index().to_string())

    # ---- filter_type_bin: factorize ----
    if "FilterType" in df.columns:
        codes, uniques = pd.factorize(df["FilterType"].fillna("missing"))
        df["filter_type_bin"] = codes.astype(int)
        logger.info("filter_type_bin mapping: %s", dict(enumerate(uniques)))
        logger.info("filter_type_bin distribution:\n%s", df["filter_type_bin"].value_counts().sort_index().to_string())

    # ---- patient_position_bin: HFS=0, FFS=1, HFP=2 ----
    if "PatientPosition" in df.columns:
        df["patient_position_bin"] = (
            df["PatientPosition"]
            .str.strip().str.upper()
            .map({"HFS": 0, "FFS": 1, "HFP": 2})
            .fillna(3)
            .astype("Int64")
        )
        logger.info("patient_position_bin distribution:\n%s",
                    df["patient_position_bin"].value_counts().sort_index().to_string())

    # ---- exposure_mod_bin: factorize ----
    if "ExposureModulationType" in df.columns:
        codes, uniques = pd.factorize(df["ExposureModulationType"].fillna("missing"))
        df["exposure_mod_bin"] = codes.astype(int)
        logger.info("exposure_mod_bin mapping: %s", dict(enumerate(uniques)))
        logger.info("exposure_mod_bin distribution:\n%s",
                    df["exposure_mod_bin"].value_counts().sort_index().to_string())

    logger.info("init_csv_ctrate | output shape: %s", df.shape)
    logger.info("init_csv_ctrate | _bin columns: %s",
                [c for c in df.columns if c.endswith("_bin")])

    return df


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lambda_y == args.lambda_r:
        lam_tag = f"lam_{args.lambda_y}"
    else:
        lam_tag = f"lam_y_{args.lambda_y}_lam_r_{args.lambda_r}"
    log_dir = output_dir / lam_tag
    logger = setup_logger(str(log_dir))

    # Also route all child-module loggers (cmi, validation, etc.) to the same file
    import logging
    root = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            root.addHandler(handler)
            break

    try:
        _main_body(args, log_dir, logger)
    except Exception:
        logger.exception("Fatal error — see traceback below")
        raise


def _main_body(args, log_dir, logger) -> None:
    logger.info("Log path %s", str(log_dir))
    logger.info("Loading dataframe from %s", args.data_path)
    set_global_seed(args.seed)

    df_init = load_dataframe(args.data_path)
    if args.dataset_key == "mimic_cxr_chest":
        df = init_csv_mimic_cxr(df_init, logger)
    elif args.dataset_key == "rsna_mammo":
        df = init_csv_rsna(df_init, logger)
    elif args.dataset_key == "rsna_mammo_mirai":
        df = init_csv_rsna_mirai(df_init, logger, gt_col=args.gt_col)
    elif args.dataset_key == "vindr_mammo":
        df = init_csv_vindr(df_init, logger)
    elif args.dataset_key == "chexpert":
        df = init_csv_chexpert(df_init, logger)
    elif args.dataset_key == "nih":
        df = init_csv_nih(df_init, logger)
    elif args.dataset_key == "ctrate":
        df = init_csv_ctrate(df_init, logger)
    else:
        df = df_init

    # ------------------------------------------------------------------
    # If --cmi-estimator=discrete, binarize R once, globally, using the
    # F1-optimal threshold learned on the validation split.  The binarized
    # column is stored as `<prob_col>_bin` and used by the discrete Stage 2
    # CMI estimator.  T_a evaluation continues to use the continuous R.
    # ------------------------------------------------------------------
    prob_col_bin = f"{args.prob_col}_bin"
    if args.cmi_estimator == "discrete":
        # Pick val rows to compute the threshold from
        val_mask = df[args.split_col] == args.val_split
        if int(val_mask.sum()) < 20:
            raise ValueError(
                f"--cmi-estimator=discrete requires a usable validation split to learn "
                f"the F1-optimal R threshold; got n_val={int(val_mask.sum())}."
            )
        val_y = pd.to_numeric(df.loc[val_mask, args.gt_col], errors="coerce").fillna(0).astype(int).values
        val_p = pd.to_numeric(df.loc[val_mask, args.prob_col], errors="coerce").fillna(0.0).astype(float).values
        r_thresh = choose_threshold_f1(val_y, val_p)
        df = df.copy()
        df[prob_col_bin] = (pd.to_numeric(df[args.prob_col], errors="coerce").fillna(0.0).astype(float).values >= r_thresh).astype(int)
        logger.info(
            "R binarization | threshold (F1-optimal on val)=%.3f | prob_col_bin=%s | "
            "n_R_bin=1 overall=%d / %d (%.3f)",
            r_thresh, prob_col_bin,
            int((df[prob_col_bin] == 1).sum()), len(df),
            float((df[prob_col_bin] == 1).mean()),
        )
    else:
        r_thresh = None

    subgroup_cols = [x.strip() for x in args.subgroup_cols.split(",") if x.strip()]

    for subgroup_col in subgroup_cols:
        subgroup_out = log_dir / subgroup_col
        subgroup_out.mkdir(parents=True, exist_ok=True)
        with timed(logger, f"Full pipeline for subgroup {subgroup_col}"):
            run_for_subgroup(df, args, subgroup_col, log_dir, logger)

    logger.info("Finished all runs. Results saved to %s", log_dir)


if __name__ == "__main__":
    main()
