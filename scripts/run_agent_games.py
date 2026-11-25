"""Agents against simulated opponents (Table 7, Figure 5). Owner: Selina.

Needs results/lstm_final.pt from run_lstm_experiments.py. Zunaer's simulated opponents
and online-only agent are added to OPPONENTS / ONLINE below once rps/opponents.py exists;
until then only the Random sanity check runs.

Usage: python scripts/run_agent_games.py --games 50
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rps import agents, game, lstm  # noqa: E402

# name -> function(seed) -> Agent. Zunaer: add Biased, Cycler, Win-stay/lose-shift, Switcher.
OPPONENTS = {"random": lambda s: agents.RandomAgent(s)}
# Zunaer: replace with the online-only agent, e.g. lambda s: opponents.OnlineOnlyAgent(window=30, seed=s)
ONLINE = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/lstm_final.pt")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    model, ckpt = lstm.load_model(args.model)
    k, fs = ckpt["k"], ckpt["feature_set"]

    players = {"static": lambda s: agents.StaticAgent(model, k, fs, seed=s)}
    if ONLINE is not None:
        players["online-only"] = ONLINE
        players["blended"] = lambda s: agents.BlendedAgent(
            agents.StaticAgent(model, k, fs, seed=s), ONLINE(s + 10_000), w=args.w, seed=s)

    rows, per_round = [], []
    for pname, make_p in players.items():
        for oname, make_o in OPPONENTS.items():
            res = game.play_many(make_p, make_o, n_games=args.games)
            res["agent1"], res["agent2"] = pname, oname
            rows.append(game.summarize_games(res)); per_round.append(res)
            print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "agent_vs_opponents.csv"), index=False)
    pd.concat(per_round).to_csv(os.path.join(args.out, "agent_vs_opponents_rounds.csv"), index=False)


if __name__ == "__main__":
    main()
