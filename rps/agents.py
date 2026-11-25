"""Selina's agents: static (pretrained LSTM) and blended (LSTM + online-only).

Zunaer's online-only agent and simulated opponents go in rps/opponents.py and
subclass rps.game.Agent; BlendedAgent accepts any Agent as its `online` part.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import encode_round, outcome_of
from .game import Agent, UNIFORM


class RandomAgent(Agent):
    """Uniform random player. Sanity check: every agent should average about 0 against it."""
    name = "random"

    def act(self, history):
        return int(self.rng.integers(3))


class StaticAgent(Agent):
    """Uses the pretrained LSTM with no player-specific update.

    The LSTM was trained to predict a *human* from the human's point of view, so the
    window is built with the opponent in the "human" slot and this agent in the "bot" slot.
    Before k rounds have been played it predicts uniformly (so it plays at random).
    """
    name = "static"

    def __init__(self, model, k: int, feature_set: str, seed: int | None = None):
        super().__init__(seed)
        self.model, self.k, self.feature_set = model, k, feature_set
        self.model.eval()

    def predict(self, history):
        if len(history) < self.k:
            return UNIFORM.copy()
        window = np.stack([encode_round(opp, me, outcome_of(opp, me), self.feature_set)
                           for me, opp in history[-self.k:]])
        with torch.no_grad():
            logits = self.model(torch.from_numpy(window[None]))
        return torch.softmax(logits, 1).numpy()[0].astype(np.float64)


class BlendedAgent(Agent):
    """p = w * p_static + (1 - w) * p_online.

    w = 1 is the static agent and w = 0 is the online-only agent, so the blend only
    adds value if a middle w beats both endpoints. Choose w on validation data.
    """
    name = "blended"

    def __init__(self, static: Agent, online: Agent, w: float = 0.5, seed: int | None = None):
        super().__init__(seed)
        assert 0.0 <= w <= 1.0
        self.static, self.online, self.w = static, online, w

    def reset(self):
        self.static.reset(); self.online.reset()

    def predict(self, history):
        p = self.w * self.static.predict(history) + (1 - self.w) * self.online.predict(history)
        return p / p.sum()

    def update(self, my_move, opp_move):
        self.static.update(my_move, opp_move); self.online.update(my_move, opp_move)


def blend_probs(p_static: np.ndarray, p_online: np.ndarray, w: float) -> np.ndarray:
    """Same blend for offline scoring on held-out human rounds (the "Blended" row of Table 4)."""
    p = w * p_static + (1 - w) * p_online
    return p / p.sum(1, keepdims=True)
