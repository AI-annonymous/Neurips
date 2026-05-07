from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class PosteriorConfig:
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    batch_size: int = 1024
    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    num_workers: int = 0


class MLPPosterior(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: Tuple[int, ...], dropout: float):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        if input_dim == 0:
            self.bias_logits = nn.Parameter(torch.zeros(num_classes))
            self.net = None
        else:
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers += [nn.Linear(prev, num_classes)]
            self.net = nn.Sequential(*layers)
            self.bias_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim == 0:
            n = x.shape[0]
            return self.bias_logits.unsqueeze(0).expand(n, -1)
        return self.net(x)


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(float(init_temp))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp_min(1e-4)



def numpy_to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)



def labels_to_tensor(y: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(y, dtype=torch.long, device=device)



def train_posterior_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    num_classes: int,
    device: torch.device,
    config: PosteriorConfig,
    verbose: bool = False,
) -> MLPPosterior:
    model = MLPPosterior(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    ds_train = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.long),
    )
    dl_train = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    if X_val is not None and y_val is not None and len(X_val) > 0:
        Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        yv = torch.as_tensor(y_val, dtype=torch.long, device=device)
    else:
        Xv = yv = None

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_left = config.patience

    for epoch in range(config.max_epochs):
        model.train()
        total = 0.0
        n_seen = 0
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * xb.shape[0]
            n_seen += xb.shape[0]

        if Xv is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(Xv)
                val_loss = float(criterion(val_logits, yv).item())
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_left = config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        else:
            # no validation split, keep final state
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    return model



def fit_temperature_scaler(
    model: MLPPosterior,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    device: torch.device,
    max_iter: int = 100,
) -> TemperatureScaler:
    scaler = TemperatureScaler().to(device)
    if len(X_cal) == 0:
        return scaler.eval()

    Xc = torch.as_tensor(X_cal, dtype=torch.float32, device=device)
    yc = torch.as_tensor(y_cal, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(Xc)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        scaled = scaler(logits)
        loss = criterion(scaled, yc)
        loss.backward()
        return loss

    optimizer.step(closure)
    scaler.eval()
    return scaler


@torch.no_grad()
def predict_proba(
    model: MLPPosterior,
    X: np.ndarray,
    device: torch.device,
    scaler: Optional[TemperatureScaler] = None,
    batch_size: int = 4096,
) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, model.num_classes), dtype=np.float32)
    model.eval()
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        if scaler is not None:
            logits = scaler(logits)
        probs = torch.softmax(logits, dim=-1)
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


@torch.no_grad()
def predict_logits(
    model: MLPPosterior,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((0, model.num_classes), dtype=np.float32)
    model.eval()
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        out.append(logits.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
