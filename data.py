from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit



def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p)
    raise ValueError(f"Unsupported dataframe format: {suffix}")



def save_dataframe(df: pd.DataFrame, path: str) -> None:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    elif suffix in {".pkl", ".pickle"}:
        df.to_pickle(p)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported dataframe format: {suffix}")



def split_by_column(
    df: pd.DataFrame,
    split_col: str = "split",
    train_name: str = "train",
    val_name: str = "validate",
    test_name: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df[split_col] == train_name].reset_index(drop=True).copy()
    val_df = df.loc[df[split_col] == val_name].reset_index(drop=True).copy()
    test_df = df.loc[df[split_col] == test_name].reset_index(drop=True).copy()
    return train_df, val_df, test_df



def stratified_train_val_split(
    df: pd.DataFrame,
    stratify_col: str,
    val_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) == 0:
        return df.copy(), df.copy()
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    y = df[stratify_col].astype(str).values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    idx = np.arange(len(df))
    train_idx, val_idx = next(splitter.split(idx, y))
    return df.iloc[train_idx].reset_index(drop=True).copy(), df.iloc[val_idx].reset_index(drop=True).copy()


@dataclass
class BlockSpec:
    name: str
    block_type: str  # "categorical" or "continuous"
    dim: int
    categories: Optional[List] = None


class OneHotBlockEncoder:
    """Block-level encoder.

    Each semantic variable becomes one block. Categorical variables are one-hot encoded.
    Continuous variables are passed through as a single standardized scalar.
    This makes it easy to gate entire semantic variables in the differentiable search.
    """

    def __init__(self, categorical_cols: Sequence[str], continuous_cols: Optional[Sequence[str]] = None):
        self.categorical_cols = list(categorical_cols)
        self.continuous_cols = list(continuous_cols or [])
        self.block_specs: Dict[str, BlockSpec] = {}
        self._cat_maps: Dict[str, Dict] = {}
        self._cont_stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> "OneHotBlockEncoder":
        for col in self.categorical_cols:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                vals = pd.Series(series).fillna(-1).astype(int).tolist()
            else:
                vals = pd.Series(series).fillna("__MISSING__").astype(str).tolist()
            cats = list(pd.Index(vals).drop_duplicates())
            mapping = {c: i for i, c in enumerate(cats)}
            self._cat_maps[col] = mapping
            self.block_specs[col] = BlockSpec(name=col, block_type="categorical", dim=len(cats), categories=list(cats))

        for col in self.continuous_cols:
            arr = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(0.0).values
            mean = float(np.mean(arr))
            std = float(np.std(arr) + 1e-8)
            self._cont_stats[col] = (mean, std)
            self.block_specs[col] = BlockSpec(name=col, block_type="continuous", dim=1, categories=None)
        return self

    @property
    def block_names(self) -> List[str]:
        return list(self.block_specs.keys())

    def transform_block(self, df: pd.DataFrame, col: str) -> np.ndarray:
        spec = self.block_specs[col]
        if spec.block_type == "categorical":
            mapping = self._cat_maps[col]
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = pd.Series(df[col]).fillna(-1).astype(int).tolist()
            else:
                vals = pd.Series(df[col]).fillna("__MISSING__").astype(str).tolist()
            idx = np.array([mapping.get(v, 0) for v in vals], dtype=np.int64)
            out = np.zeros((len(df), spec.dim), dtype=np.float32)
            out[np.arange(len(df)), idx] = 1.0
            return out
        mean, std = self._cont_stats[col]
        arr = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(mean).values
        arr = ((arr - mean) / std).reshape(-1, 1).astype(np.float32)
        return arr

    def transform_blocks(self, df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, np.ndarray]:
        return {c: self.transform_block(df, c) for c in cols}

    def concat_blocks(self, block_dict: Dict[str, np.ndarray], cols: Sequence[str]) -> np.ndarray:
        if not cols:
            return np.zeros((next(iter(block_dict.values())).shape[0], 0), dtype=np.float32) if block_dict else np.zeros((0, 0), dtype=np.float32)
        mats = [block_dict[c] for c in cols]
        return np.concatenate(mats, axis=1).astype(np.float32)



def build_input_matrix(
    df: pd.DataFrame,
    encoder: OneHotBlockEncoder,
    metadata_cols: Sequence[str],
    include_y: bool = False,
    y_col: Optional[str] = None,
    include_r: bool = False,
    prob_col: Optional[str] = None,
) -> np.ndarray:
    blocks = encoder.transform_blocks(df, metadata_cols)
    X_meta = encoder.concat_blocks(blocks, metadata_cols) if metadata_cols else np.zeros((len(df), 0), dtype=np.float32)
    extras: List[np.ndarray] = [X_meta]
    if include_y:
        assert y_col is not None
        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
        y_oh = np.zeros((len(df), 2), dtype=np.float32)
        y = np.clip(y, 0, 1)
        y_oh[np.arange(len(df)), y] = 1.0
        extras.append(y_oh)
    if include_r:
        assert prob_col is not None
        r = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).astype(float).values.reshape(-1, 1).astype(np.float32)
        extras.append(r)
    if len(extras) == 1:
        return extras[0]
    return np.concatenate(extras, axis=1).astype(np.float32)
