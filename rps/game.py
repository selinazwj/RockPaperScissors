"""Game framework: the agent interface, expected-reward move choice and the game loop.

Owner: Selina. Every agent (Selina's static/blended agents, Zunaer's online-only agent
and simulated opponents) subclasses `Agent`, so any two of them can play each other.

Moves are ints: 0 = rock, 1 = paper, 2 = scissors. Each agent sees the history from its
own point of view as a list of (my_move, opp_move) pairs for rounds already played.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIFORM = np.full(3, 1 / 3)


def reward(a: int, b: int) -> int:
    """+1 if a beats b, 0 for a tie, -1 if a loses."""
    return 0 if a == b else (1 if (a - b) % 3 == 1 else -1)


def expected_rewards(p_opp: np.ndarray) -> np.ndarray:
    """E[reward(a)] = P(opponent plays the move a beats) - P(opponent plays the move that beats a)."""
    return np.array([p_opp[(a - 1) % 3] - p_opp[(a + 1) % 3] for a in range(3)])


def best_response(p_opp: np.ndarray, rng: np.random.Generator) -> int:
    """Move with the highest expected reward; ties broken at random (uniform p -> random move)."""
    er = expected_rewards(p_opp)
    return int(rng.choice(np.flatnonzero(np.isclose(er, er.max()))))


class Agent:
    """Interface for every player.

    predict(history) -> probabilities of the opponent's next move (rock, paper, scissors)
    act(history)     -> this agent's move; default is the best response to predict()
    update(my_move, opp_move) -> called after both moves of a round are revealed
    reset()          -> forget everything about the current opponent (between games)
    """
    name = "agent"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def predict(self, history: list[tuple[int, int]]) -> np.ndarray:
        return UNIFORM.copy()

    def act(self, history: list[tuple[int, int]]) -> int:
        return best_response(self.predict(history), self.rng)

    def update(self, my_move: int, opp_move: int) -> None:
        pass


def play_game(agent1: Agent, agent2: Agent, n_rounds: int = 300) -> pd.DataFrame:
    """Play one game. Both agents choose from history up to t-1, then both see the result.

    Neither agent can read the other's current move: both act() calls happen before
    either update() call.
    """
    agent1.reset(); agent2.reset()
    h1, h2, rows = [], [], []
    for t in range(1, n_rounds + 1):
        m1, m2 = agent1.act(list(h1)), agent2.act(list(h2))
        agent1.update(m1, m2); agent2.update(m2, m1)
        h1.append((m1, m2)); h2.append((m2, m1))
        rows.append((t, m1, m2, reward(m1, m2)))
    return pd.DataFrame(rows, columns=["round", "move1", "move2", "reward1"])


def play_many(make_agent1, make_agent2, n_games: int = 50, n_rounds: int = 300,
              seed: int = 0) -> pd.DataFrame:
    """Repeat games with fresh agents and different seeds. make_agent*(seed) -> Agent."""
    games = []
    for g in range(n_games):
        a1, a2 = make_agent1(seed + 2 * g), make_agent2(seed + 2 * g + 1)
        df = play_game(a1, a2, n_rounds)
        df.insert(0, "game", g)
        df["agent1"], df["agent2"] = a1.name, a2.name
        games.append(df)
    return pd.concat(games, ignore_index=True)


def summarize_games(results: pd.DataFrame) -> dict:
    """Average reward per round for agent 1, win/tie/loss rates, and spread across games."""
    per_game = results.groupby("game").reward1.mean()
    r = results.reward1
    return {"agent1": results.agent1.iloc[0], "agent2": results.agent2.iloc[0],
            "games": results.game.nunique(), "mean_reward": float(r.mean()),
            "sd_across_games": float(per_game.std(ddof=1)) if len(per_game) > 1 else 0.0,
            "win": float((r == 1).mean()), "tie": float((r == 0).mean()),
            "loss": float((r == -1).mean())}
