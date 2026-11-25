"""E1 (history length), E2 (feature set) on validation, then E3 on test. Owner: Selina.

Usage (from the repo root, locally or in Colab):
    python scripts/run_lstm_experiments.py                 # full run
    python scripts/run_lstm_experiments.py --quick         # 1 seed, few epochs, to check it runs

Writes to results/:
    lstm_validation.csv     every (k, feature set, seed) run on validation  -> Tables 2 and 5, Figure 3
    lstm_history.csv        per-epoch losses of every run                   -> appendix loss curves
    lstm_test_rounds.csv    per-round test results of the final LSTM        -> Table 4, bootstrap, E4
    lstm_final.pt           final model checkpoint (used by the agents)
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rps import data, evaluate, lstm  # noqa: E402


def windows_for(rounds, split, k, fs, which):
    part = rounds[rounds.game_id.map(split) == which]
    return data.build_windows(part, k, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=data.DATA_URL)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = [0] if args.quick else list(range(args.seeds))
    train_kw = dict(hidden_size=args.hidden, max_epochs=3 if args.quick else 50)

    rounds = data.load_rounds(args.data)
    split = data.temporary_split(rounds)          # swap for Zunaer's split when merged
    print(f"{rounds.game_id.nunique()} games;",
          pd.Series(split).value_counts().to_dict())

    runs, histories = [], []

    def run(k, fs, experiment):
        Xtr, ytr, _ = windows_for(rounds, split, k, fs, "train")
        Xva, yva, _ = windows_for(rounds, split, k, fs, "val")
        for seed in seeds:
            model, hist = lstm.train_lstm(Xtr, ytr, Xva, yva, seed=seed, **train_kw)
            res = evaluate.summarize(lstm.predict_proba(model, Xva), yva)
            runs.append(dict(experiment=experiment, k=k, feature_set=fs, seed=seed,
                             epochs=len(hist), val_accuracy=res["accuracy"],
                             val_cross_entropy=res["cross_entropy"], n_val=res["n"]))
            histories.extend([dict(experiment=experiment, k=k, feature_set=fs, seed=seed, **h)
                          for h in hist])
            print(runs[-1])

    # E1: history length with feature set C
    for k in (3, 5, 10):
        run(k, "C", "E1")
    table = pd.DataFrame(runs)
    best_k = int(table.groupby("k").val_cross_entropy.mean().idxmin())
    # E2: feature sets at the best k (C at best_k is reused from E1)
    for fs in ("A", "B"):
        run(best_k, fs, "E2")
    table = pd.DataFrame(runs)
    table.to_csv(os.path.join(args.out, "lstm_validation.csv"), index=False)
    pd.DataFrame(histories).to_csv(os.path.join(args.out, "lstm_history.csv"), index=False)

    sel = table[table.k == best_k].groupby("feature_set").val_cross_entropy.mean()
    best_fs = sel.idxmin()
    print(f"selected k={best_k}, feature set={best_fs} (by validation cross-entropy)")

    # E3: retrain the selected configuration and score the test set once
    Xtr, ytr, _ = windows_for(rounds, split, best_k, best_fs, "train")
    Xva, yva, _ = windows_for(rounds, split, best_k, best_fs, "val")
    Xte, yte, mte = windows_for(rounds, split, best_k, best_fs, "test")
    test_rows, summaries = [], []
    for seed in seeds:
        model, _ = lstm.train_lstm(Xtr, ytr, Xva, yva, seed=seed, **train_kw)
        p = lstm.predict_proba(model, Xte)
        summaries.append(evaluate.summarize(p, yte, f"LSTM seed {seed}"))
        test_rows.append(evaluate.per_round(mte, p, yte, f"lstm_seed{seed}"))
        if seed == seeds[0]:
            lstm.save_model(model, os.path.join(args.out, "lstm_final.pt"),
                            k=best_k, feature_set=best_fs)
    pd.concat(test_rows).to_csv(os.path.join(args.out, "lstm_test_rounds.csv"), index=False)
    s = pd.DataFrame(summaries)
    print(s.to_string(index=False))
    print(json.dumps({"k": best_k, "feature_set": best_fs,
                      "test_accuracy_mean": s.accuracy.mean(), "test_accuracy_sd": s.accuracy.std(),
                      "test_ce_mean": s.cross_entropy.mean()}, indent=2))


if __name__ == "__main__":
    main()
