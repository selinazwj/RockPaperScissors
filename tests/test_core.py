"""Sanity checks. Run with: python -m pytest tests  (or: python tests/test_core.py)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rps import agents, data, evaluate, game, lstm  # noqa: E402


def toy_rounds(n_games=4, n_rounds=40, timeout_at=None, seed=0):
    """Fake games where the human always plays the move that beats their previous move."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_games):
        h = int(rng.integers(3))
        for r in range(1, n_rounds + 1):
            b = int(rng.integers(3))
            hm = h if r != timeout_at else data.MISSING
            o = data.outcome_of(h, b) if hm != data.MISSING else data.MISSING
            rows.append((f"g{g}", "toy", r, hm, b, o))
            h = (h + 1) % 3
    return pd.DataFrame(rows, columns=["game_id", "bot_strategy", "round_index",
                                       "human_move", "bot_move", "outcome"])


def test_beats_and_outcome():
    R, P, S = 0, 1, 2
    assert data.beats(P, R) and data.beats(R, S) and data.beats(S, P)
    assert not data.beats(R, P)
    assert data.outcome_of(R, R) == data.OUTCOME_TO_INT["tie"]
    assert game.reward(P, R) == 1 and game.reward(R, P) == -1 and game.reward(S, S) == 0


def test_best_response():
    rng = np.random.default_rng(0)
    assert game.best_response(np.array([1.0, 0, 0]), rng) == 1      # opponent rock -> paper
    assert game.best_response(np.array([0, 0, 1.0]), rng) == 0      # opponent scissors -> rock
    er = game.expected_rewards(np.array([0.6, 0.25, 0.15]))
    assert np.isclose(er[1], 0.6 - 0.15)


def test_windows_shape_and_same_rounds():
    rounds = toy_rounds()
    for k in (3, 5, 10):
        X, y, meta = data.build_windows(rounds, k, "C", min_round=11)
        assert X.shape[1:] == (k, 9)
        assert meta.round_index.min() == 11                # same scored rounds for every k
        assert len(y) == 4 * 30


def test_windows_skip_timeouts():
    rounds = toy_rounds(n_games=1, timeout_at=20)
    X, y, meta = data.build_windows(rounds, 3, "A", min_round=11)
    bad = set(range(20, 24))                               # target 20 or window covering 20
    assert not bad & set(meta.round_index)
    assert len(y) == 30 - 4


def test_metrics():
    y = np.array([0, 1, 2])
    p = np.full((3, 3), 1 / 3)
    assert np.isclose(evaluate.cross_entropy(p, y), np.log(3))
    p2 = np.eye(3)
    assert evaluate.accuracy(p2, y) == 1.0
    assert evaluate.confusion_matrix(p2, y).trace() == 3


def test_lstm_learns_simple_pattern():
    rounds = toy_rounds(n_games=20, n_rounds=60)
    X, y, _ = data.build_windows(rounds, 3, "A", min_round=4)
    model, hist = lstm.train_lstm(X, y, X, y, hidden_size=16, lr=0.01, max_epochs=40, seed=0)
    assert evaluate.accuracy(lstm.predict_proba(model, X), y) > 0.95


def test_game_loop_has_no_peeking():
    class Peeker(game.Agent):
        name = "peeker"
        def __init__(self, seed=None):
            super().__init__(seed); self.seen = []
        def act(self, history):
            self.seen.append(len(history)); return 0
    a, b = Peeker(), Peeker()
    df = game.play_game(a, b, n_rounds=10)
    assert a.seen == list(range(10)) and b.seen == list(range(10))
    assert (df.reward1 == 0).all()


def test_random_vs_random_is_fair():
    res = game.play_many(lambda s: agents.RandomAgent(s), lambda s: agents.RandomAgent(s),
                         n_games=20, n_rounds=300)
    assert abs(game.summarize_games(res)["mean_reward"]) < 0.05


def test_static_agent_beats_its_pattern():
    """A static agent trained on 'human plays the move that beats their last move' should
    beat an opponent who does exactly that."""
    rounds = toy_rounds(n_games=20, n_rounds=60)
    X, y, _ = data.build_windows(rounds, 3, "C", min_round=4)
    model, _ = lstm.train_lstm(X, y, X, y, hidden_size=16, lr=0.01, max_epochs=40, seed=0)

    class Cycler(game.Agent):
        name = "cycler"
        def act(self, history):
            return 0 if not history else (history[-1][0] + 1) % 3

    res = game.play_many(lambda s: agents.StaticAgent(model, 3, "C", seed=s),
                         lambda s: Cycler(s), n_games=5, n_rounds=100)
    assert game.summarize_games(res)["mean_reward"] > 0.8


def test_blend_endpoints():
    ps, po = np.array([[0.7, 0.2, 0.1]]), np.array([[0.1, 0.1, 0.8]])
    assert np.allclose(agents.blend_probs(ps, po, 1.0), ps)
    assert np.allclose(agents.blend_probs(ps, po, 0.0), po)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
