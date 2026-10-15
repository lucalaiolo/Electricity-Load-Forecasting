"""Neural forecasters in PyTorch: an MLP on the flat window and an LSTM over the hourly sequence."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..config import HOURS


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, width = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, HOURS))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMForecaster(nn.Module):
    """LSTM encoder over the last week of hourly inputs; an MLP head reads its final
    hidden state together with the calendar of the target day and emits 24 values."""

    def __init__(self, n_seq: int, n_static: int, hidden: int = 64, head: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, HOURS)
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], dim=1))


def _batches(n: int, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


def train_network(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    val_inputs: tuple[torch.Tensor, ...] | None = None,
    val_target: torch.Tensor | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    patience: int = 25,
    seed: int = 0,
) -> tuple[nn.Module, dict]:
    """Mini-batch AdamW on the mean squared error.

    With validation data the weights of the epoch with the lowest validation MAE are
    restored (early stopping); without it the model trains for exactly ``epochs``.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train_mse": [], "val_mae": []}
    best = (np.inf, 0, copy.deepcopy(model.state_dict()))
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for idx in _batches(len(target), batch_size, gen):
            opt.zero_grad()
            loss = loss_fn(model(*(x[idx] for x in inputs)), target[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["train_mse"].append(total / len(target))
        if val_inputs is None:
            continue
        model.eval()
        with torch.no_grad():
            val_mae = (model(*val_inputs) - val_target).abs().mean().item()
        history["val_mae"].append(val_mae)
        if val_mae < best[0]:
            best = (val_mae, epoch, copy.deepcopy(model.state_dict()))
        elif epoch - best[1] >= patience:
            break
    if val_inputs is not None:
        model.load_state_dict(best[2])
        history["best_epoch"] = best[1]
    return model, history


class Standardizer:
    def fit(self, X: np.ndarray) -> Standardizer:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean


class NeuralForecaster:
    """Seed ensemble of MLPs or LSTMs with input/target scaling handled internally.

    Inputs are tuples: ``(X_wide,)`` for the MLP, ``(X_seq, X_static)`` for the LSTM,
    where X_seq is already scaled and the 2-D blocks are standardised here.
    """

    def __init__(self, kind: str = "mlp", seeds=(0, 1, 2, 3, 4), arch: dict | None = None, **train_kw):
        self.kind, self.seeds, self.arch, self.train_kw = kind, seeds, arch or {}, train_kw

    def _prep(self, inputs: tuple[np.ndarray, ...]) -> tuple[torch.Tensor, ...]:
        out = []
        for x, scaler in zip(inputs, self.scalers):
            x = scaler.transform(x) if scaler is not None else x
            out.append(torch.as_tensor(np.asarray(x, dtype=np.float32)))
        return tuple(out)

    def _new_model(self, inputs: tuple[np.ndarray, ...]) -> nn.Module:
        if self.kind == "mlp":
            return MLP(inputs[0].shape[1], **self.arch)
        return LSTMForecaster(inputs[0].shape[2], inputs[1].shape[1], **self.arch)

    def fit(self, inputs, Y, val_inputs=None, Y_val=None, epochs: int | None = None) -> NeuralForecaster:
        self.scalers = [None if x.ndim == 3 else Standardizer().fit(np.asarray(x, float)) for x in inputs]
        self.y_scaler = Standardizer().fit(np.asarray(Y, float))
        X = self._prep(inputs)
        T = torch.as_tensor(self.y_scaler.transform(Y).astype(np.float32))
        kw = dict(self.train_kw)
        if epochs is not None:
            kw["epochs"] = epochs
        if val_inputs is not None:
            kw["val_inputs"] = self._prep(val_inputs)
            kw["val_target"] = torch.as_tensor(self.y_scaler.transform(Y_val).astype(np.float32))
        self.models, self.histories = [], []
        for seed in self.seeds:
            torch.manual_seed(seed)
            model, hist = train_network(self._new_model(inputs), X, T, seed=seed, **kw)
            self.models.append(model.eval())
            self.histories.append(hist)
        return self

    @property
    def best_epochs(self) -> list[int]:
        return [h["best_epoch"] for h in self.histories]

    def predict(self, inputs) -> np.ndarray:
        X = self._prep(inputs)
        with torch.no_grad():
            Z = np.mean([m(*X).numpy() for m in self.models], axis=0)
        return self.y_scaler.inverse(Z)
