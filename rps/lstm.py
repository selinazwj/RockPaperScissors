"""LSTM next-move predictor. Owner: Selina.

One architecture only (per proposal feedback): LSTM -> linear layer -> 3 logits.
Hyperparameters are chosen on validation cross-entropy; the test set is touched once.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.config = dict(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (batch, k, features)
        _, (h, _) = self.lstm(x)
        return self.out(self.drop(h[-1]))                     # logits: (batch, 3)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    parts = [torch.softmax(model(torch.from_numpy(X[i:i + batch_size])), 1).numpy()
             for i in range(0, len(X), batch_size)]
    return np.concatenate(parts).astype(np.float64)


def train_lstm(X_train, y_train, X_val, y_val, hidden_size=32, num_layers=1, dropout=0.0,
               lr=1e-3, batch_size=256, max_epochs=50, patience=5, seed=0, verbose=False):
    """Train with Adam + cross-entropy and stop early on validation cross-entropy.

    Returns the model with the best validation weights and a per-epoch history
    (use the history for the loss-curve figure in the appendix).
    """
    set_seed(seed)
    model = LSTMPredictor(X_train.shape[2], hidden_size, num_layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X_train), torch.from_numpy(y_train)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)
    gen = torch.Generator().manual_seed(seed)

    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(Xv)
            val_loss = loss_fn(logits, yv).item()
            val_acc = (logits.argmax(1) == yv).float().mean().item()
        history.append(dict(epoch=epoch, train_loss=total / len(Xt), val_loss=val_loss,
                            val_acc=val_acc))
        if verbose:
            print(history[-1])
        if val_loss < best - 1e-4:
            best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def save_model(model: LSTMPredictor, path: str, **extra) -> None:
    torch.save({"config": model.config, "state": model.state_dict(), **extra}, path)


def load_model(path: str):
    ckpt = torch.load(path, weights_only=False)
    model = LSTMPredictor(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt
