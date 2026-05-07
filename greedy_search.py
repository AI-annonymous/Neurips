from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .cmi import (
    CMIConfig,
    CMIResult,
    DiscreteCMIConfig,
    estimate_cmi_stage1,
    estimate_cmi_stage1_discrete,
    estimate_cmi_stage2,
    estimate_cmi_stage2_discrete,
)
from .data import OneHotBlockEncoder


@dataclass
class SearchConfig:
    lambda_penalty: float = 0.01
    tolerance: float = 1e-4
    stage1_exact_discrete: bool = False
    stage1_laplace_alpha: float = 0.0
    stage2_exact_discrete: bool = False
    stage2_laplace_alpha: float = 0.0
    prob_col_bin: Optional[str] = None  # Required when stage2_exact_discrete=True
    # Greedy lookahead per stage.  When beam_size_stage{N} > 1, the forward
    # step for that stage also considers adding k-tuples of unselected
    # variables (k = 2 .. beam_size_stage{N}) after single-variable forward
    # stalls.  Mitigates the non-monotonicity of conditional MI when path-
    # blocking variables only reduce CMI in groups (e.g., {Q_1, Q_2, Q_3}
    # together blocks A->R but no proper subset does).
    #
    # Default 1 = standard greedy (no beam).
    # beam_size_stage1: usually 1 — Stage 1's I(A;Y|V) is well-behaved.
    # beam_size_stage2: often 2-3 needed — Stage 2's I(A;R|Y,V) frequently
    #   exhibits collider-like non-monotonicity.
    beam_size_stage1: int = 1
    beam_size_stage2: int = 1


@dataclass
class SearchStep:
    action: str
    subset: Tuple[str, ...]
    objective: float
    mi: float
    ce0: float
    ce1: float


@dataclass
class GreedySearchResult:
    selected: List[str]
    objective: float
    mi: float
    ce0: float
    ce1: float
    history: List[SearchStep] = field(default_factory=list)
    cache: Dict[Tuple[str, ...], Tuple[float, CMIResult]] = field(default_factory=dict)


class GreedySubsetSearcher:
    def __init__(
        self,
        df_search: pd.DataFrame,
        encoder: OneHotBlockEncoder,
        a_col: str,
        y_col: str,
        prob_col: str,
        device,
        cmi_config: CMIConfig,
        search_config: SearchConfig,
        logger,
    ):
        self.df_search = df_search.reset_index(drop=True).copy()
        self.encoder = encoder
        self.a_col = a_col
        self.y_col = y_col
        self.prob_col = prob_col
        self.device = device
        self.cmi_config = cmi_config
        self.search_config = search_config
        self.logger = logger
        self._cache_stage1: Dict[Tuple[str, ...], Tuple[float, CMIResult]] = {}
        self._cache_stage2: Dict[Tuple[str, ...], Tuple[float, CMIResult]] = {}

    def _objective_stage1(self, subset: Sequence[str]) -> Tuple[float, CMIResult]:
        key = tuple(sorted(subset))
        if key not in self._cache_stage1:
            if self.search_config.stage1_exact_discrete:
                result = estimate_cmi_stage1_discrete(
                    self.df_search,
                    key,
                    a_col=self.a_col,
                    y_col=self.y_col,
                    config=DiscreteCMIConfig(laplace_alpha=self.search_config.stage1_laplace_alpha),
                )
            else:
                result = estimate_cmi_stage1(
                    self.df_search,
                    key,
                    self.encoder,
                    a_col=self.a_col,
                    y_col=self.y_col,
                    device=self.device,
                    config=self.cmi_config,
                )
            obj = result.mi + self.search_config.lambda_penalty * len(key)
            self._cache_stage1[key] = (obj, result)
        return self._cache_stage1[key]

    def _objective_stage2(self, subset: Sequence[str]) -> Tuple[float, CMIResult]:
        key = tuple(sorted(subset))
        if key not in self._cache_stage2:
            if self.search_config.stage2_exact_discrete:
                if not self.search_config.prob_col_bin:
                    raise ValueError("stage2_exact_discrete requires prob_col_bin to be set on SearchConfig.")
                result = estimate_cmi_stage2_discrete(
                    self.df_search,
                    key,
                    a_col=self.a_col,
                    y_col=self.y_col,
                    prob_col_bin=self.search_config.prob_col_bin,
                    config=DiscreteCMIConfig(laplace_alpha=self.search_config.stage2_laplace_alpha),
                )
            else:
                result = estimate_cmi_stage2(
                    self.df_search,
                    key,
                    self.encoder,
                    a_col=self.a_col,
                    y_col=self.y_col,
                    prob_col=self.prob_col,
                    device=self.device,
                    config=self.cmi_config,
                )
            obj = result.mi + self.search_config.lambda_penalty * len(key)
            self._cache_stage2[key] = (obj, result)
        return self._cache_stage2[key]

    def _run_single_stage(self, candidates: Sequence[str], stage: str) -> GreedySearchResult:
        if stage not in {"stage1", "stage2"}:
            raise ValueError(stage)
        objective_fn = self._objective_stage1 if stage == "stage1" else self._objective_stage2
        selected: List[str] = []
        history: List[SearchStep] = []

        current_obj, current_res = objective_fn(selected)
        history.append(SearchStep("init", tuple(selected), current_obj, current_res.mi, current_res.ce0, current_res.ce1))

        while True:
            best_j = None
            best_obj = current_obj
            best_res = current_res
            for j in candidates:
                if j in selected:
                    continue
                trial = list(selected) + [j]
                obj, res = objective_fn(trial)
                if obj < best_obj:
                    best_j = j
                    best_obj = obj
                    best_res = res
            if best_j is None or (current_obj - best_obj) <= self.search_config.tolerance:
                break
            selected.append(best_j)
            current_obj, current_res = best_obj, best_res
            history.append(SearchStep("add", tuple(sorted(selected)), current_obj, current_res.mi, current_res.ce0, current_res.ce1))
            self.logger.info("[%s] add %s -> subset=%s obj=%.6f mi=%.6f", stage, best_j, selected, current_obj, current_res.mi)

        # Beam-search forward step (k > 1).  Runs only when single-step
        # forward stalled at a local minimum and beam_size > 1.  Considers
        # adding all k-tuples of unselected variables for k = 2 .. beam_size.
        # This addresses the non-monotonicity of conditional MI on subsets
        # where path-blocking variables only reduce CMI in groups (e.g., a
        # set of acquisition confounders that jointly d-separate A from R
        # but individually open colliders).
        # Pick the beam size for the current stage.
        if stage == "stage1":
            beam_size = max(1, int(self.search_config.beam_size_stage1))
        else:
            beam_size = max(1, int(self.search_config.beam_size_stage2))
        if beam_size >= 2:
            while True:
                remaining = [j for j in candidates if j not in selected]
                if len(remaining) < 2:
                    break
                best_combo = None
                best_obj = current_obj
                best_res = current_res
                # Try combinations of size 2..beam_size (capped at len(remaining))
                for k in range(2, min(beam_size, len(remaining)) + 1):
                    for combo in combinations(remaining, k):
                        trial = list(selected) + list(combo)
                        obj, res = objective_fn(trial)
                        if obj < best_obj:
                            best_combo = combo
                            best_obj = obj
                            best_res = res
                if best_combo is None or (current_obj - best_obj) <= self.search_config.tolerance:
                    break
                selected.extend(best_combo)
                current_obj, current_res = best_obj, best_res
                history.append(SearchStep(
                    "add_beam", tuple(sorted(selected)), current_obj,
                    current_res.mi, current_res.ce0, current_res.ce1,
                ))
                self.logger.info(
                    "[%s] add_beam %s -> subset=%s obj=%.6f mi=%.6f",
                    stage, list(best_combo), sorted(selected),
                    current_obj, current_res.mi,
                )

        while True:
            best_remove = None
            best_obj = current_obj
            best_res = current_res
            for j in list(selected):
                trial = [x for x in selected if x != j]
                obj, res = objective_fn(trial)
                if obj < best_obj:
                    best_remove = j
                    best_obj = obj
                    best_res = res
            if best_remove is None or (current_obj - best_obj) <= self.search_config.tolerance:
                break
            selected.remove(best_remove)
            current_obj, current_res = best_obj, best_res
            history.append(SearchStep("remove", tuple(sorted(selected)), current_obj, current_res.mi, current_res.ce0, current_res.ce1))
            self.logger.info("[%s] remove %s -> subset=%s obj=%.6f mi=%.6f", stage, best_remove, selected, current_obj, current_res.mi)

        while True:
            best_swap = None
            best_obj = current_obj
            best_res = current_res
            for j in list(selected):
                for l in candidates:
                    if l in selected:
                        continue
                    trial = [x for x in selected if x != j] + [l]
                    obj, res = objective_fn(trial)
                    if obj < best_obj:
                        best_swap = (j, l)
                        best_obj = obj
                        best_res = res
            if best_swap is None or (current_obj - best_obj) <= self.search_config.tolerance:
                break
            j, l = best_swap
            selected = [x for x in selected if x != j] + [l]
            selected = sorted(selected)
            current_obj, current_res = best_obj, best_res
            history.append(SearchStep("swap", tuple(sorted(selected)), current_obj, current_res.mi, current_res.ce0, current_res.ce1))
            self.logger.info("[%s] swap %s->%s -> subset=%s obj=%.6f mi=%.6f", stage, j, l, selected, current_obj, current_res.mi)

        cache = self._cache_stage1 if stage == "stage1" else self._cache_stage2
        return GreedySearchResult(
            selected=sorted(selected),
            objective=float(current_obj),
            mi=float(current_res.mi),
            ce0=float(current_res.ce0),
            ce1=float(current_res.ce1),
            history=history,
            cache=cache,
        )

    def search_stage1(self, candidates: Sequence[str]) -> GreedySearchResult:
        return self._run_single_stage(candidates, stage="stage1")

    def search_stage2(self, candidates: Sequence[str]) -> GreedySearchResult:
        return self._run_single_stage(candidates, stage="stage2")