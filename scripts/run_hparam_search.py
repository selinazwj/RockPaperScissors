"""Small hyperparameter search around the selected LSTM (k=10, feature set C). Owner: Selina.

Each variant changes one setting from the v1 model (hidden 32, 1 layer, dropout 0, lr 0.001)
and is scored on validation only, mean of 3 seeds. Output: results/lstm_hparams.csv (Table 6).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rps import data, evaluate, lstm  # noqa: E402

BASE = dict(hidden_size=32, num_layers=1, dropout=0.0, lr=1e-3)
VARIANTS = {
    "v1 (selected so far)": {},
    "v2: hidden 64": dict(hidden_size=64),
    "v3: learning rate 0.003": dict(lr=3e-3),
    "v4: 2 layers + dropout 0.2": dict(num_layers=2, dropout=0.2),
}


def main(path=data.DATA_URL, out="results", k=10, fs="C", seeds=(0, 1, 2)):
    rounds = data.load_rounds(path)
    split = data.temporary_split(rounds)
    part = lambda w: rounds[rounds.game_id.map(split) == w]
    Xtr, ytr, _ = data.build_windows(part("train"), k, fs)
    Xva, yva, _ = data.build_windows(part("val"), k, fs)
    rows = []
    for name, change in VARIANTS.items():
        for seed in seeds:
            model, hist = lstm.train_lstm(Xtr, ytr, Xva, yva, seed=seed, **{**BASE, **change})
            s = evaluate.summarize(lstm.predict_proba(model, Xva), yva)
            rows.append(dict(variant=name, seed=seed, epochs=len(hist),
                             val_accuracy=s["accuracy"], val_cross_entropy=s["cross_entropy"]))
            print(rows[-1])
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "lstm_hparams.csv"), index=False)
    print(df.groupby("variant", sort=False)[["val_accuracy", "val_cross_entropy", "epochs"]]
            .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:2])
