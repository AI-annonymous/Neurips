"""Plotting helpers for the appendix figures of the validation diagnostic suite.

Reads sidecar artifacts written by ``write_sidecar_artifacts`` and produces
three appendix figures per (control, subgroup):

    1. Love plot of per-V*-block SMD before vs after weighting.
    2. Reliability curve for the weight model P̂(A=a|V*).
    3. Violin / histogram of the normalized weights.

Filename convention (after the ``__`` separator change in
weight_diagnostics):
    diagnostics__{control}__{col_name}-{value}__{kind}.{ext}
For backward compatibility the legacy single-underscore format with
subgroup token ``A-{value}`` (synthetic-only) is also accepted.

Plot title format:
    {Method} | Subgroup: {col_name}={value}
where Method is inferred from the sidecar directory's grandparent
(``exhaustive`` / ``greedy`` / ``gated`` / combinations like
``gated+exhaustive``); falls back to the control name if the layout
doesn't match.

Run as a script:

    python -m causal_analysis.diagnostic_plots \
        --sidecar-dir output/.../diagnostics \
        --fig-dir     output/.../figures
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib required for plotting helpers") from e


_log = logging.getLogger(__name__)

_METHOD_DISPLAY = {
    "exhaustive":        "Exhaustive",
    "greedy":            "Greedy",
    "gated":             "Gated",
    "gated+exhaustive":  "Gated+Exhaustive",
    "greedy+exhaustive": "Greedy+Exhaustive",
    "exhaustive+greedy": "Exhaustive+Greedy",
    "gated+greedy":      "Gated+Greedy",
    "greedy+gated":      "Greedy+Gated",
    "exhaustive+gated":  "Exhaustive+Gated",
}


def _parse_filename(path: str) -> Optional[Tuple[str, str, str]]:
    """Parse ``diagnostics__{control}__{sg}__{kind}.{ext}`` into pieces.

    Returns ``(control, sg_token, kind)`` where ``sg_token`` is
    ``{col_name}-{value}`` and ``kind`` is one of ``smd | reliability |
    weights``. Falls back to the legacy single-underscore format
    ``diagnostics_{control}_A-{value}_{kind}.{ext}`` when the new
    delimiter is absent.
    """
    base = os.path.basename(path)

    # New format: __ separators throughout.
    m = re.match(
        r"^diagnostics__(?P<control>.+?)__(?P<sg>.+?)__"
        r"(?P<kind>smd|reliability|weights)\.(?:csv|npy)$",
        base,
    )
    if m is not None:
        return m["control"], m["sg"], m["kind"]

    # Legacy format: single underscore, subgroup must be `A-{value}`.
    m = re.match(
        r"^diagnostics_(?P<control>.+?)_(?P<sg>A-[^_]+)_"
        r"(?P<kind>smd|reliability|weights)\.(?:csv|npy)$",
        base,
    )
    if m is not None:
        return m["control"], m["sg"], m["kind"]

    return None


def _method_from_path(sidecar_path: str) -> Optional[str]:
    """Infer search method from grandparent directory name."""
    sidecar_dir = os.path.dirname(os.path.abspath(sidecar_path))
    parent = os.path.basename(os.path.dirname(sidecar_dir))
    return _METHOD_DISPLAY.get(parent.lower())


def _format_subgroup(sg_token: str) -> str:
    """``A-1`` -> ``A=1``; ``age_bin-3`` -> ``age_bin=3``."""
    if "-" not in sg_token:
        return sg_token
    col, _, value = sg_token.rpartition("-")
    return f"{col}={value}"


def _build_title(sidecar_path: str, control: str, sg_token: str) -> str:
    """``{Method or Control} | Subgroup: {col}={value}``."""
    method = _method_from_path(sidecar_path)
    head = method if method is not None else control
    return f"{head} | Subgroup: {_format_subgroup(sg_token)}"


def _safe_filename_token(s: str) -> str:
    """Replace filesystem-unfriendly characters in a sidecar token."""
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
#  Plot primitives
# ---------------------------------------------------------------------------

def love_plot(smd_csv: str, ax=None, max_smd_threshold: float = 0.1):
    df = pd.read_csv(smd_csv)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 0.4 * max(len(df), 4) + 1))
    df = df.sort_values("smd_before_max", ascending=True)
    y = np.arange(len(df))
    ax.scatter(df["smd_before_max"], y, marker="o", label="before weighting")
    ax.scatter(df["smd_after_max"],  y, marker="x", label="after weighting")
    for yi, (b, a) in enumerate(zip(df["smd_before_max"], df["smd_after_max"])):
        ax.plot([b, a], [yi, yi], color="gray", alpha=0.4, linewidth=0.8)
    ax.axvline(max_smd_threshold, color="red", linestyle="--", linewidth=0.8,
               label=f"|SMD|={max_smd_threshold} (Austin 2009)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["block"].values)
    ax.set_xlabel("max |Standardized Mean Difference| within block")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def reliability_curve(reliability_csv: str, ax=None):
    df = pd.read_csv(reliability_csv)
    df = df[df["count"] > 0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    sizes = 30 + 200 * df["count"] / max(df["count"].max(), 1)
    ax.scatter(df["conf"], df["acc"], s=sizes, alpha=0.7, edgecolor="k")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"Mean predicted $\hat P(A=a|V^*)$ in bin")
    ax.set_ylabel(r"Empirical frequency of $A=a$ in bin")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def weight_violin(weights_npy: str, ax=None, log_scale: bool = True,
                  x_label: Optional[str] = None):
    w = np.load(weights_npy)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if log_scale and np.std(w) >= 1e-3:
        ax.set_yscale("log")
    parts = ax.violinplot(w, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="mean = 1")
    p99 = float(np.percentile(w, 99))
    ax.axhline(p99, color="orange", linestyle=":", linewidth=0.8,
               label=f"p99 = {p99:.2f}")
    ax.set_xticks([1])
    if x_label is None:
        x_label = (os.path.basename(weights_npy)
                   .replace("__weights.npy", "")
                   .replace("_weights.npy", ""))
    ax.set_xticklabels([x_label])
    ax.set_ylabel(r"Normalized weight $\tilde w_i$ (mean 1)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

def make_all_appendix_figures(sidecar_dir: str, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)

    # Glob both new (__) and legacy (_) layouts.
    smd_files = sorted(set(
        glob.glob(os.path.join(sidecar_dir, "diagnostics__*__smd.csv")) +
        glob.glob(os.path.join(sidecar_dir, "diagnostics_*_smd.csv"))
    ))
    rel_files = sorted(set(
        glob.glob(os.path.join(sidecar_dir, "diagnostics__*__reliability.csv")) +
        glob.glob(os.path.join(sidecar_dir, "diagnostics_*_reliability.csv"))
    ))
    w_files = sorted(set(
        glob.glob(os.path.join(sidecar_dir, "diagnostics__*__weights.npy")) +
        glob.glob(os.path.join(sidecar_dir, "diagnostics_*_weights.npy"))
    ))

    n_total = len(smd_files) + len(rel_files) + len(w_files)
    n_parsed = 0
    n_written = 0

    def _emit(files, plot_fn, kind_name: str):
        nonlocal n_parsed, n_written
        for f in files:
            info = _parse_filename(f)
            if info is None:
                _log.warning(
                    "diagnostic_plots: could not parse filename '%s' (skipped)",
                    os.path.basename(f),
                )
                continue
            n_parsed += 1
            ctrl, sg, _kind = info
            try:
                if kind_name == "weights":
                    method = _method_from_path(f)
                    x_label = method if method is not None else ctrl
                    ax = plot_fn(f, x_label=f"{x_label} ({_format_subgroup(sg)})")
                else:
                    ax = plot_fn(f)
                ax.set_title(_build_title(f, ctrl, sg))
                plt.tight_layout()
                out_name = (f"{kind_name}__{_safe_filename_token(ctrl)}__"
                            f"{_safe_filename_token(sg)}.pdf")
                plt.savefig(os.path.join(fig_dir, out_name))
                plt.close()
                n_written += 1
            except Exception as exc:
                _log.warning(
                    "diagnostic_plots: failed to render '%s': %s",
                    os.path.basename(f), exc,
                )
                plt.close("all")

    _emit(smd_files, love_plot,         "loveplot")
    _emit(rel_files, reliability_curve, "reliability")
    _emit(w_files,   weight_violin,     "weights")

    if n_total > 0 and n_written == 0:
        _log.warning(
            "diagnostic_plots: found %d sidecar files in %s but produced 0 "
            "figures. Check filename format and _parse_filename regex.",
            n_total, sidecar_dir,
        )
    else:
        _log.info(
            "diagnostic_plots: parsed %d/%d sidecar files, wrote %d figures to %s",
            n_parsed, n_total, n_written, fig_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--fig-dir",     required=True)
    args = p.parse_args()
    make_all_appendix_figures(args.sidecar_dir, args.fig_dir)