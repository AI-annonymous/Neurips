from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

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
from .greedy_search import GreedySearchResult, SearchConfig, SearchStep


class ExhaustiveSubsetSearcher:
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

    @staticmethod
    def _all_subsets(candidates: Sequence[str]):
        ordered = list(dict.fromkeys(candidates))
        for r in range(len(ordered) + 1):
            for subset in combinations(ordered, r):
                yield tuple(sorted(subset))

    def _run_single_stage(self, candidates: Sequence[str], stage: str) -> GreedySearchResult:
        objective_fn = self._objective_stage1 if stage == 'stage1' else self._objective_stage2
        history: List[SearchStep] = []
        best_subset: Tuple[str, ...] = tuple()
        best_obj = float('inf')
        best_res: CMIResult | None = None

        total_subsets = 2 ** len(candidates)
        self.logger.info('[%s-exhaustive] enumerating %d subsets from %d candidates: %s',
                         stage, total_subsets, len(candidates), list(candidates))

        from tqdm import tqdm
        for subset in tqdm(self._all_subsets(candidates), total=total_subsets,
                           desc=f'{stage} exhaustive', unit='subset'):
            obj, res = objective_fn(subset)
            history.append(SearchStep('eval', subset, float(obj), float(res.mi), float(res.ce0), float(res.ce1)))
            if obj < best_obj - self.search_config.tolerance:
                best_subset = subset
                best_obj = float(obj)
                best_res = res

        if best_res is None:
            raise RuntimeError(f'No subsets were evaluated for {stage}.')

        self.logger.info('[%s-exhaustive] evaluated=%d best_subset=%s obj=%.6f mi=%.6f', stage, len(history), list(best_subset), best_obj, best_res.mi)
        cache = self._cache_stage1 if stage == 'stage1' else self._cache_stage2
        return GreedySearchResult(
            selected=list(best_subset),
            objective=float(best_obj),
            mi=float(best_res.mi),
            ce0=float(best_res.ce0),
            ce1=float(best_res.ce1),
            history=history,
            cache=cache,
        )

    def search_stage1(self, candidates: Sequence[str]) -> GreedySearchResult:
        return self._run_single_stage(candidates, stage='stage1')

    def search_stage2(self, candidates: Sequence[str]) -> GreedySearchResult:
        return self._run_single_stage(candidates, stage='stage2')
