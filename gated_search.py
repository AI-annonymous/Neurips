from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from torch.func import functional_call  # type: ignore
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore

from .cmi import CMIConfig, estimate_cmi_stage1, estimate_cmi_stage2
from .data import OneHotBlockEncoder, build_input_matrix
from .models import MLPPosterior


@dataclass
class GatedSearchConfig:
    lambda_penalty: float = 0.01
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.0
    batch_size: int = 1024
    epochs: int = 30
    lr_head: float = 1e-3
    lr_gate: float = 5e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    random_state: int = 42
    hc_temperature: float = 0.5
    hc_gamma: float = -0.1
    hc_zeta: float = 1.1
    threshold_grid: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class GatedSearchResult:
    selected: List[str]
    probabilities: Dict[str, float]
    objective: float
    mi: float = 0.0
    ce0: float = 0.0
    ce1: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)
    threshold_scores: List[Dict[str, float]] = field(default_factory=list)


class HardConcreteGate(nn.Module):
    def __init__(self, n_blocks: int, temperature: float, gamma: float, zeta: float):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_blocks))
        self.temperature = temperature
        self.gamma = gamma
        self.zeta = zeta

    def sample(self, training: bool = True) -> torch.Tensor:
        if training:
            u = torch.rand_like(self.alpha).clamp_(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.alpha) / self.temperature)
            z_tilde = s * (self.zeta - self.gamma) + self.gamma
            return z_tilde.clamp(0.0, 1.0)
        return self.probabilities()

    def probabilities(self) -> torch.Tensor:
        offset = self.temperature * np.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.alpha - offset)


class _StageGatedModule(nn.Module):
    def __init__(
        self,
        block_dims: Sequence[int],
        extra0_dim: int,
        extra1_dim: int,
        num_classes: int,
        cfg: GatedSearchConfig,
    ):
        super().__init__()
        self.block_dims = list(block_dims)
        self.n_blocks = len(self.block_dims)
        self.gates = HardConcreteGate(self.n_blocks, cfg.hc_temperature, cfg.hc_gamma, cfg.hc_zeta)
        input_meta_dim = int(sum(self.block_dims))
        self.head0 = MLPPosterior(input_meta_dim + extra0_dim, num_classes, cfg.hidden_dims, cfg.dropout)
        self.head1 = MLPPosterior(input_meta_dim + extra1_dim, num_classes, cfg.hidden_dims, cfg.dropout)

    def _apply_gates(self, blocks: List[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        if not blocks:
            return torch.zeros((z.shape[0], 0), device=z.device)
        gated = [blocks[j] * z[j] for j in range(len(blocks))]
        return torch.cat(gated, dim=1)

    def forward_head0(self, blocks: List[torch.Tensor], extra0: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        x_meta = self._apply_gates(blocks, z)
        x = x_meta if extra0 is None else torch.cat([x_meta, extra0], dim=1)
        return self.head0(x)

    def forward_head1(self, blocks: List[torch.Tensor], extra1: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        x_meta = self._apply_gates(blocks, z)
        x = x_meta if extra1 is None else torch.cat([x_meta, extra1], dim=1)
        return self.head1(x)



def _make_split_indices(n: int, y: np.ndarray, val_fraction: float, random_state: int):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
    idx = np.arange(n)
    tr_idx, val_idx = next(splitter.split(idx, y))
    return tr_idx, val_idx



def _stack_blocks(df: pd.DataFrame, encoder: OneHotBlockEncoder, cols: Sequence[str], device: torch.device) -> List[torch.Tensor]:
    blocks = []
    for c in cols:
        arr = encoder.transform_block(df, c)
        blocks.append(torch.as_tensor(arr, dtype=torch.float32, device=device))
    return blocks



def _build_extras_stage1(df: pd.DataFrame, y_col: str, device: torch.device):
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
    y = np.clip(y, 0, 1)
    y_oh = np.zeros((len(df), 2), dtype=np.float32)
    y_oh[np.arange(len(df)), y] = 1.0
    return None, torch.as_tensor(y_oh, dtype=torch.float32, device=device)



def _build_extras_stage2(df: pd.DataFrame, y_col: str, prob_col: str, device: torch.device):
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
    y = np.clip(y, 0, 1)
    y_oh = np.zeros((len(df), 2), dtype=np.float32)
    y_oh[np.arange(len(df)), y] = 1.0
    y_oh_t = torch.as_tensor(y_oh, dtype=torch.float32, device=device)
    r = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).astype(float).values.reshape(-1, 1)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=device)
    return y_oh_t, torch.cat([y_oh_t, r_t], dim=1)



def _gather_blocks(blocks: List[torch.Tensor], idx: torch.Tensor) -> List[torch.Tensor]:
    return [b[idx] for b in blocks]



def _adapt_params(model: nn.Module, loss: torch.Tensor, lr: float) -> Dict[str, torch.Tensor]:
    params = {name: p for name, p in model.named_parameters()}
    grads = torch.autograd.grad(loss, list(params.values()), create_graph=True, allow_unused=False)
    adapted = {name: p - lr * g for (name, p), g in zip(params.items(), grads)}
    return adapted



def _compute_val_objective_stage(
    module: _StageGatedModule,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    blocks: List[torch.Tensor],
    extra0: Optional[torch.Tensor],
    extra1: Optional[torch.Tensor],
    labels: torch.Tensor,
    cfg: GatedSearchConfig,
) -> torch.Tensor:
    criterion = nn.CrossEntropyLoss()

    train_blocks = _gather_blocks(blocks, train_idx)
    val_blocks = _gather_blocks(blocks, val_idx)
    extra0_train = None if extra0 is None else extra0[train_idx]
    extra1_train = None if extra1 is None else extra1[train_idx]
    extra0_val = None if extra0 is None else extra0[val_idx]
    extra1_val = None if extra1 is None else extra1[val_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]

    z_train = module.gates.sample(training=True)
    train_logits0 = module.forward_head0(train_blocks, extra0_train, z_train)
    train_logits1 = module.forward_head1(train_blocks, extra1_train, z_train)
    train_loss0 = criterion(train_logits0, y_train)
    train_loss1 = criterion(train_logits1, y_train)

    adapted0 = _adapt_params(module.head0, train_loss0, cfg.lr_head)
    adapted1 = _adapt_params(module.head1, train_loss1, cfg.lr_head)

    z_val = module.gates.sample(training=False)
    x_meta_val = module._apply_gates(val_blocks, z_val)
    x0_val = x_meta_val if extra0_val is None else torch.cat([x_meta_val, extra0_val], dim=1)
    x1_val = x_meta_val if extra1_val is None else torch.cat([x_meta_val, extra1_val], dim=1)

    logits0_val = functional_call(module.head0, adapted0, (x0_val,))
    logits1_val = functional_call(module.head1, adapted1, (x1_val,))
    val_loss0 = criterion(logits0_val, y_val)
    val_loss1 = criterion(logits1_val, y_val)
    penalty = cfg.lambda_penalty * module.gates.probabilities().sum()
    return val_loss0 - val_loss1 + penalty



def _fit_stage_gated(
    df_search: pd.DataFrame,
    subset_candidates: Sequence[str],
    encoder: OneHotBlockEncoder,
    a_col: str,
    y_col: str,
    prob_col: str,
    stage: str,
    device: torch.device,
    cmi_config: CMIConfig,
    cfg: GatedSearchConfig,
    logger,
) -> GatedSearchResult:
    if stage not in {"stage1", "stage2"}:
        raise ValueError(stage)
    if len(subset_candidates) == 0:
        return GatedSearchResult(selected=[], probabilities={}, objective=0.0, history=[], threshold_scores=[])

    mapping_vals = sorted(pd.Series(df_search[a_col]).dropna().astype(int if pd.api.types.is_numeric_dtype(df_search[a_col]) else str).unique().tolist())
    mapping = {v: i for i, v in enumerate(mapping_vals)}
    labels_np = np.array([mapping.get(v, 0) for v in df_search[a_col].tolist()], dtype=np.int64)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
    num_classes = len(mapping)

    tr_idx_np, val_idx_np = _make_split_indices(len(df_search), labels_np, cfg.val_fraction, cfg.random_state)
    tr_idx = torch.as_tensor(tr_idx_np, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(val_idx_np, dtype=torch.long, device=device)

    blocks = _stack_blocks(df_search, encoder, subset_candidates, device=device)
    block_dims = [b.shape[1] for b in blocks]
    if stage == "stage1":
        extra0, extra1 = _build_extras_stage1(df_search, y_col, device)
        extra0_dim = 0
        extra1_dim = 2
    else:
        extra0, extra1 = _build_extras_stage2(df_search, y_col, prob_col, device)
        extra0_dim = 2
        extra1_dim = 3

    module = _StageGatedModule(block_dims, extra0_dim, extra1_dim, num_classes, cfg).to(device)
    opt_head0 = torch.optim.AdamW(module.head0.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    opt_head1 = torch.optim.AdamW(module.head1.parameters(), lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    opt_gate = torch.optim.Adam([module.gates.alpha], lr=cfg.lr_gate)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(torch.as_tensor(tr_idx_np, dtype=torch.long)), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.as_tensor(val_idx_np, dtype=torch.long)), batch_size=cfg.batch_size, shuffle=True)
    history: List[Dict[str, float]] = []

    for epoch in tqdm(range(cfg.epochs), desc=f"{stage}-gated", leave=False):
        module.train()
        val_iter = iter(val_loader)
        train_loss0_epoch = 0.0
        train_loss1_epoch = 0.0
        gate_obj_epoch = 0.0
        n_batches = 0

        for (batch_idx_cpu,) in train_loader:
            batch_idx = batch_idx_cpu.to(device)
            try:
                (val_batch_idx_cpu,) = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                (val_batch_idx_cpu,) = next(val_iter)
            val_batch_idx = val_batch_idx_cpu.to(device)

            batch_blocks = _gather_blocks(blocks, batch_idx)
            batch_labels = labels[batch_idx]
            batch_extra0 = None if extra0 is None else extra0[batch_idx]
            batch_extra1 = None if extra1 is None else extra1[batch_idx]

            # Update heads on train batch
            z = module.gates.sample(training=True).detach()
            opt_head0.zero_grad(set_to_none=True)
            logits0 = module.forward_head0(batch_blocks, batch_extra0, z)
            loss0 = criterion(logits0, batch_labels)
            loss0.backward()
            opt_head0.step()

            opt_head1.zero_grad(set_to_none=True)
            logits1 = module.forward_head1(batch_blocks, batch_extra1, z)
            loss1 = criterion(logits1, batch_labels)
            loss1.backward()
            opt_head1.step()

            # Gate update with one-step unrolling
            opt_gate.zero_grad(set_to_none=True)
            gate_obj = _compute_val_objective_stage(
                module,
                train_idx=batch_idx,
                val_idx=val_batch_idx,
                blocks=blocks,
                extra0=extra0,
                extra1=extra1,
                labels=labels,
                cfg=cfg,
            )
            gate_obj.backward()
            opt_gate.step()

            train_loss0_epoch += float(loss0.item())
            train_loss1_epoch += float(loss1.item())
            gate_obj_epoch += float(gate_obj.item())
            n_batches += 1

        probs = module.gates.probabilities().detach().cpu().numpy().tolist()
        hist = {
            "epoch": float(epoch + 1),
            "train_loss0": train_loss0_epoch / max(1, n_batches),
            "train_loss1": train_loss1_epoch / max(1, n_batches),
            "gate_objective": gate_obj_epoch / max(1, n_batches),
        }
        for name, p in zip(subset_candidates, probs):
            hist[f"pi::{name}"] = float(p)
        history.append(hist)
        logger.info(
            "[%s-gated] epoch=%03d loss0=%.4f loss1=%.4f gate=%.4f selected~=%s",
            stage,
            epoch + 1,
            hist["train_loss0"],
            hist["train_loss1"],
            hist["gate_objective"],
            {k: round(v, 3) for k, v in zip(subset_candidates, probs)},
        )

    pi = module.gates.probabilities().detach().cpu().numpy()
    prob_map = {name: float(p) for name, p in zip(subset_candidates, pi.tolist())}

    threshold_scores: List[Dict[str, float]] = []
    tau_candidates = sorted(set(list(cfg.threshold_grid) + [round(float(x), 3) for x in pi.tolist()]))
    best_obj = float("inf")
    best_subset: List[str] = []
    best_tau = 0.5
    best_mi = 0.0
    best_ce0 = 0.0
    best_ce1 = 0.0

    for tau in tau_candidates:
        subset = [name for name, p in prob_map.items() if p >= tau]
        if stage == "stage1":
            res = estimate_cmi_stage1(df_search, subset, encoder, a_col=a_col, y_col=y_col, device=device, config=cmi_config)
        else:
            res = estimate_cmi_stage2(df_search, subset, encoder, a_col=a_col, y_col=y_col, prob_col=prob_col, device=device, config=cmi_config)
        obj = res.mi + cfg.lambda_penalty * len(subset)
        threshold_scores.append({
            "tau": float(tau),
            "subset_size": float(len(subset)),
            "objective": float(obj),
            "mi": float(res.mi),
            "ce0": float(res.ce0),
            "ce1": float(res.ce1),
            "subset": subset,
        })
        if obj < best_obj:
            best_obj = float(obj)
            best_subset = subset
            best_tau = float(tau)
            best_mi = float(res.mi)
            best_ce0 = float(res.ce0)
            best_ce1 = float(res.ce1)

    logger.info("[%s-gated] best threshold=%.3f -> subset=%s objective=%.6f mi=%.6f", stage, best_tau, best_subset, best_obj, best_mi)
    return GatedSearchResult(
        selected=best_subset,
        probabilities=prob_map,
        objective=best_obj,
        mi=best_mi,
        ce0=best_ce0,
        ce1=best_ce1,
        history=history,
        threshold_scores=threshold_scores,
    )


class GatedSearcher:
    def __init__(
        self,
        df_search: pd.DataFrame,
        encoder: OneHotBlockEncoder,
        a_col: str,
        y_col: str,
        prob_col: str,
        device,
        cmi_config: CMIConfig,
        gated_config: GatedSearchConfig,
        logger,
    ):
        self.df_search = df_search.reset_index(drop=True).copy()
        self.encoder = encoder
        self.a_col = a_col
        self.y_col = y_col
        self.prob_col = prob_col
        self.device = device
        self.cmi_config = cmi_config
        self.gated_config = gated_config
        self.logger = logger

    def search_stage1(self, candidates: Sequence[str]) -> GatedSearchResult:
        return _fit_stage_gated(
            self.df_search,
            candidates,
            self.encoder,
            a_col=self.a_col,
            y_col=self.y_col,
            prob_col=self.prob_col,
            stage="stage1",
            device=self.device,
            cmi_config=self.cmi_config,
            cfg=self.gated_config,
            logger=self.logger,
        )

    def search_stage2(self, candidates: Sequence[str]) -> GatedSearchResult:
        return _fit_stage_gated(
            self.df_search,
            candidates,
            self.encoder,
            a_col=self.a_col,
            y_col=self.y_col,
            prob_col=self.prob_col,
            stage="stage2",
            device=self.device,
            cmi_config=self.cmi_config,
            cfg=self.gated_config,
            logger=self.logger,
        )
