"""Loading the Brockbank & Vul RPS data and turning it into LSTM training windows.

Owner: Selina (sequence building). The cleaning and the participant split belong to
Zunaer; `load_rounds` and `temporary_split` below are minimal stand-ins so the LSTM
code can run now. Replace them with Zunaer's versions once they are merged, keeping
the same output columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MOVES = ["rock", "paper", "scissors"]
MOVE_TO_INT = {m: i for i, m in enumerate(MOVES)}
OUTCOMES = ["win", "tie", "loss"]          # from the human's point of view
OUTCOME_TO_INT = {o: i for i, o in enumerate(OUTCOMES)}
MISSING = -1                                # timeout ("none") or unknown

FEATURE_SETS = {
    "A": ("human",),                        # human moves only           -> 3 values
    "B": ("human", "bot"),                  # + bot moves                -> 6 values
    "C": ("human", "bot", "outcome"),       # + human outcome            -> 9 values
}
DATA_URL = ("https://raw.githubusercontent.com/erik-brockbank/hm_rps_public/main/"
            "data/human_data/rps_human_trial_data.csv")


def beats(a: int, b: int) -> bool:
    """True if move a beats move b (rock 0 > scissors 2, paper 1 > rock 0, scissors 2 > paper 1)."""
    return (a - b) % 3 == 1


def outcome_of(a: int, b: int) -> int:
    """Outcome index for the player who played a against b."""
    if a == b:
        return OUTCOME_TO_INT["tie"]
    return OUTCOME_TO_INT["win"] if beats(a, b) else OUTCOME_TO_INT["loss"]


def load_rounds(path: str = DATA_URL) -> pd.DataFrame:
    """One row per round: game_id, bot_strategy, round_index, human_move, bot_move, outcome.

    TEMPORARY cleaning (to be replaced by Zunaer's): keep games where the human has
    exactly 300 rounds, and for duplicate sona_survey_code keep the earliest game.
    Timeouts stay in the table with human_move = -1 so that windows can skip them.
    """
    raw = pd.read_csv(path)
    human = raw[raw.is_bot == 0]
    bot = raw[raw.is_bot == 1][["game_id", "round_index", "player_move"]]
    bot = bot.rename(columns={"player_move": "bot_move"})
    df = human.merge(bot, on=["game_id", "round_index"], how="inner")
    assert not df.duplicated(["game_id", "round_index"]).any(), "more than one pair per round"

    n_rounds = df.groupby("game_id").round_index.transform("count")
    df = df[n_rounds == 300]
    first_ts = df.groupby("game_id").round_begin_ts.transform("min")
    df = df.assign(first_ts=first_ts)
    keep = (df.drop_duplicates("game_id").sort_values("first_ts")
              .drop_duplicates("sona_survey_code").game_id)
    df = df[df.game_id.isin(keep)]

    out = pd.DataFrame({
        "game_id": df.game_id.values,
        "bot_strategy": df.bot_strategy.values,
        "round_index": df.round_index.astype(int).values,
        "human_move": df.player_move.map(MOVE_TO_INT).fillna(MISSING).astype(int).values,
        "bot_move": df.bot_move.map(MOVE_TO_INT).fillna(MISSING).astype(int).values,
        "outcome": df.player_outcome.map(OUTCOME_TO_INT).fillna(MISSING).astype(int).values,
    })
    return out.sort_values(["game_id", "round_index"]).reset_index(drop=True)


def temporary_split(rounds: pd.DataFrame, seed: int = 340,
                    fractions=(0.70, 0.15, 0.15)) -> dict[str, str]:
    """game_id -> 'train' / 'val' / 'test', stratified by bot strategy. Stand-in for Zunaer's split."""
    rng = np.random.default_rng(seed)
    split = {}
    games = rounds.drop_duplicates("game_id")[["game_id", "bot_strategy"]]
    for _, group in games.groupby("bot_strategy"):
        ids = group.game_id.to_numpy().copy()
        rng.shuffle(ids)
        n_tr = int(round(fractions[0] * len(ids)))
        n_va = int(round(fractions[1] * len(ids)))
        for i, g in enumerate(ids):
            split[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return split


def encode_round(human: int, bot: int, outcome: int, feature_set: str) -> np.ndarray:
    parts = []
    for name in FEATURE_SETS[feature_set]:
        v = np.zeros(3, dtype=np.float32)
        v[{"human": human, "bot": bot, "outcome": outcome}[name]] = 1.0
        parts.append(v)
    return np.concatenate(parts)


def feature_dim(feature_set: str) -> int:
    return 3 * len(FEATURE_SETS[feature_set])


def build_windows(rounds: pd.DataFrame, k: int, feature_set: str, min_round: int = 11):
    """Turn rounds into (X, y, meta) for next-move prediction.

    X[i] holds rounds t-k .. t-1 of one game, y[i] is the human move at round t.
    Rules that keep the comparison fair and leak-free:
      * only targets with t >= min_round are used, so every model and every k is scored
        on the same rounds (min_round = 11 allows k up to 10);
      * a window is dropped if the target or any round inside it is a timeout, so a
        window never joins two rounds that were not consecutive;
      * nothing after round t-1 is used.
    """
    assert min_round > k, "min_round must leave room for k rounds of history"
    d = feature_dim(feature_set)
    X, y, meta = [], [], []
    for game_id, g in rounds.groupby("game_id", sort=False):
        g = g.sort_values("round_index")
        h, b, o = g.human_move.to_numpy(), g.bot_move.to_numpy(), g.outcome.to_numpy()
        r = g.round_index.to_numpy()
        assert (np.diff(r) == 1).all(), f"rounds not consecutive in {game_id}"
        valid = (h != MISSING) & (b != MISSING) & (o != MISSING)
        enc = np.zeros((len(g), d), dtype=np.float32)
        for i in np.flatnonzero(valid):
            enc[i] = encode_round(h[i], b[i], o[i], feature_set)
        strategy = g.bot_strategy.iloc[0]
        for t in range(len(g)):
            if r[t] < min_round or not valid[t] or not valid[t - k:t].all():
                continue
            X.append(enc[t - k:t]); y.append(h[t])
            meta.append((game_id, strategy, int(r[t]), int(h[t - 1]), int(o[t - 1])))
    meta = pd.DataFrame(meta, columns=["game_id", "bot_strategy", "round_index",
                                       "prev_human_move", "prev_outcome"])
    return np.stack(X), np.asarray(y, dtype=np.int64), meta
