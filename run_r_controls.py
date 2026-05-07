"""
Compute R-based DeAmour controls for existing search results.

Skips the search entirely — reads V_Y and V_R from command line or
existing JSON files, then computes only the specified control sets.

Usage:
    python -m causal_analysis.run_r_controls \
        --data-path <csv> \
        --output-dir <existing_output_dir>/lam_0.003 \
        --dataset-key rsna_mammo \
        --gt-col out_put_GT \
        --prob-col out_put_predict \
        --subgroup-cols site_id_bin \
        --v-y "" \
        --v-r "machine_id_bin" \
        --controls r_only,y_plus_r,r_plus_searched,y_r_plus_searched
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .data import OneHotBlockEncoder, load_dataframe, split_by_column
from .feature_groups import parse_csv_list, resolve_feature_grouping
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


def _init_df(df_init, dataset_key, logger):
    """Re-use init functions from main.py without importing the whole module."""
    if dataset_key == "mimic_cxr_chest":
        from .main import init_csv_mimic_cxr
        return init_csv_mimic_cxr(df_init, logger)
    elif dataset_key == "rsna_mammo":
        from .main import init_csv_rsna
        return init_csv_rsna(df_init, logger)
    elif dataset_key == "rsna_mammo_mirai":
        from .main import init_csv_rsna_mirai
        return init_csv_rsna_mirai(df_init, logger)
    elif dataset_key == "vindr_mammo":
        from .main import init_csv_vindr
        return init_csv_vindr(df_init, logger)
    elif dataset_key == "chexpert":
        from .main import init_csv_chexpert
        return init_csv_chexpert(df_init, logger)
    elif dataset_key == "nih":
        from .main import init_csv_nih
        return init_csv_nih(df_init, logger)
    elif dataset_key == "ctrate":
        from .main import init_csv_ctrate
        return init_csv_ctrate(df_init, logger)
    return df_init


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute R-based DeAmour controls without re-running search"
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output dir — should be the lam_X directory with existing results")
    parser.add_argument("--dataset-key", type=str, required=True)

    parser.add_argument("--split-col", type=str, default="split")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validate")
    parser.add_argument("--test-split", type=str, default="test")

    parser.add_argument("--gt-col", type=str, required=True)
    parser.add_argument("--prob-col", type=str, required=True)
    parser.add_argument("--subgroup-cols", type=str, required=True)

    # V_Y and V_R: comma-separated, or empty string for empty set
    parser.add_argument("--v-y", type=str, default=None,
                        help="Comma-separated V_Y variables. If not given, reads from existing stage1_search.json")
    parser.add_argument("--v-r", type=str, default=None,
                        help="Comma-separated V_R variables. If not given, reads from existing stage2_search.json")

    # Which search method directory to read from (for auto-loading V_Y/V_R)
    parser.add_argument("--search-method-dir", type=str, default="exhaustive",
                        help="Subdirectory name to read existing search JSONs from")

    # Which controls to compute
    parser.add_argument("--controls", type=str,
                        default="empty,r_only,y_plus_r,r_plus_searched,y_r_plus_searched",
                        help="Comma-separated control sets to compute")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")

    # Weight model config
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

    return parser.parse_args()


def _load_v_from_json(json_path: Path) -> List[str]:
    """Read 'selected' from a stage search JSON."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("selected", [])


def _build_r_control_sets(
    y_col: str, prob_col: str, searched_v: List[str], requested: List[str],
) -> Dict[str, List[str]]:
    """Build only the requested control sets."""
    all_possible = {
        "empty": [],
        "y_only": [y_col],
        "r_only": [prob_col],
        "y_plus_r": [y_col, prob_col],
        "searched": list(searched_v),
        "y_plus_searched": [y_col] + list(searched_v),
        "r_plus_searched": [prob_col] + list(searched_v),
        "y_r_plus_searched": [y_col, prob_col] + list(searched_v),
    }
    # Always include empty for Delta computation
    controls = {"empty": []}
    for name in requested:
        name = name.strip()
        if name in all_possible:
            controls[name] = all_possible[name]
        else:
            raise ValueError(f"Unknown control: {name}. Available: {list(all_possible.keys())}")
    return controls


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(output_dir), name="r_controls")

    root = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            root.addHandler(handler)
            break

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    try:
        _run(args, output_dir, device, logger)
    except Exception:
        logger.exception("Fatal error")
        raise


def _run(args, output_dir, device, logger):
    set_global_seed(args.seed)

    logger.info("Loading data from %s", args.data_path)
    df_init = load_dataframe(args.data_path)
    df = _init_df(df_init, args.dataset_key, logger)

    train_df, val_df, test_df = split_by_column(
        df, split_col=args.split_col,
        train_name=args.train_split, val_name=args.val_split, test_name=args.test_split,
    )
    logger.info("Split sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

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
    )

    requested_controls = [c.strip() for c in args.controls.split(",") if c.strip()]
    logger.info("Requested controls: %s", requested_controls)

    subgroup_cols = [x.strip() for x in args.subgroup_cols.split(",") if x.strip()]

    for subgroup_col in subgroup_cols:
        logger.info("=" * 70)
        logger.info("Subgroup: %s", subgroup_col)
        logger.info("=" * 70)

        # Resolve V_Y and V_R
        if args.v_y is not None:
            v_y = [x.strip() for x in args.v_y.split(",") if x.strip()]
        else:
            json_path = output_dir / subgroup_col / args.search_method_dir / "stage1_search.json"
            if json_path.exists():
                v_y = _load_v_from_json(json_path)
                logger.info("Loaded V_Y from %s: %s", json_path, v_y)
            else:
                logger.warning("No stage1_search.json found at %s, using V_Y=[]", json_path)
                v_y = []

        if args.v_r is not None:
            v_r = [x.strip() for x in args.v_r.split(",") if x.strip()]
        else:
            json_path = output_dir / subgroup_col / args.search_method_dir / "stage2_search.json"
            if json_path.exists():
                v_r = _load_v_from_json(json_path)
                logger.info("Loaded V_R from %s: %s", json_path, v_r)
            else:
                logger.warning("No stage2_search.json found at %s, using V_R=[]", json_path)
                v_r = []

        searched_v = list(v_y) + list(v_r)
        logger.info("V_Y=%s | V_R=%s | V*=%s", v_y, v_r, searched_v)

        # Build encoder
        all_meta_cols = sorted(set(
            v_y + v_r + [subgroup_col]
        ))
        encoder = OneHotBlockEncoder(categorical_cols=all_meta_cols).fit(df)

        # Threshold
        threshold = choose_threshold_f1(val_df[args.gt_col].values, val_df[args.prob_col].values)
        logger.info("Threshold: %.4f", threshold)

        # Metric screen
        metric_screen_df = build_metric_screen_table(
            df_test=test_df, a_col=subgroup_col, y_col=args.gt_col,
            prob_col=args.prob_col, threshold=threshold,
        )

        # Build control sets
        control_sets = _build_r_control_sets(
            args.gt_col, args.prob_col, searched_v, requested_controls,
        )
        logger.info("Control sets to compute: %s", list(control_sets.keys()))

        # Evaluate
        frames = []
        for control_name, control_cols in control_sets.items():
            with timed(logger, f"Validation {subgroup_col}:{control_name}"):
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
                )
                vdf["V_Y"] = ";".join(v_y) if v_y else ""
                vdf["V_R"] = ";".join(v_r) if v_r else ""
                vdf["V_star"] = ";".join(searched_v) if searched_v else ""
                frames.append(vdf)

        result_df = pd.concat(frames, ignore_index=True)
        result_df = add_gap_reduction(result_df, baseline_control_name="empty")

        # Save
        out_path = output_dir / subgroup_col / "r_controls_validation.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_path, index=False)
        logger.info("Saved to %s", out_path)

        # Print summary
        ok_rows = result_df[(result_df["ok"] == True) & (result_df["control"] != "empty")]
        for _, r in ok_rows.iterrows():
            ci0 = "YES" if r["ci_contains_zero"] else "NO"
            logger.info(
                "  %s | %s | %s | T_a=%+.4f | CI=[%+.4f,%+.4f] ci0=%s | Delta=%.4f",
                r["subgroup"], r["control"], r["metric"],
                r["T_a"], r["ci_lo"], r["ci_hi"], ci0,
                r.get("Delta_a_m", float("nan")),
            )

    logger.info("Done. All R-control results saved.")


if __name__ == "__main__":
    main()
