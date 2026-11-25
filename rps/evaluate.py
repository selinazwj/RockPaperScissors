"""Shared evaluation functions. Owner: Selina. Used by every model, including Zunaer's baselines.

Every predictor should produce an (N, 3) array of probabilities for rock, paper and
scissors on the same rows of `meta` (from rps.data.build_windows), so all models are
scored on identical rounds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANCE_ACCURACY = 1 / 3
CHANCE_CROSS_ENTROPY = float(np.log(3))      # 1.0986


def _check(probs: np.ndarray, y: np.ndarray) -> None:
    assert probs.ndim == 2 and probs.shape[1] == 3, "probs must be (N, 3)"
    assert len(probs) == len(y), "probs and y differ in length"
    assert np.allclose(probs.sum(1), 1, atol=1e-4), "each row of probs must sum to 1"


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    _check(probs, y)
    return float((probs.argmax(1) == y).mean())


def cross_entropy(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean of -log(probability given to the move that happened). Lower is better."""
    _check(probs, y)
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], eps, 1)).mean())


def summarize(probs: np.ndarray, y: np.ndarray, name: str = "") -> dict:
    return {"model": name, "n": len(y), "accuracy": accuracy(probs, y),
            "cross_entropy": cross_entropy(probs, y)}


def per_round(meta: pd.DataFrame, probs: np.ndarray, y: np.ndarray, name: str) -> pd.DataFrame:
    """One row per scored round, for the paired bootstrap and the per-bot breakdown (Zunaer)."""
    _check(probs, y)
    out = meta[["game_id", "bot_strategy", "round_index", "prev_outcome"]].copy()
    out["model"] = name
    out["correct"] = (probs.argmax(1) == y).astype(int)
    out["log_loss"] = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1))
    return out


def confusion_matrix(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rows = actual move, columns = predicted move (rock, paper, scissors)."""
    _check(probs, y)
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, probs.argmax(1)), 1)
    return cm
