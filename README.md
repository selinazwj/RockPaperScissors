# RockPaperScissors

DS340 final project (Selina Zhang, Zunaer Sharang): predicting human moves in
Rock-Paper-Scissors with simple baselines and an LSTM, and using the predictions in
adaptive agents. Data: Brockbank and Vul (2024),
https://github.com/erik-brockbank/hm_rps_public (downloaded automatically).

## Setup

```
pip install -r requirements.txt
python tests/test_core.py                         # sanity checks, about 30 s
python scripts/run_lstm_experiments.py --quick    # checks the pipeline end to end
python scripts/run_lstm_experiments.py            # E1, E2, E3 with 3 seeds
python scripts/run_agent_games.py --games 50      # agents vs simulated opponents
```

In Colab: `!git clone <this repo>`, `%cd RockPaperScissors`, then run the same commands.

## Layout and owners

| Path | What | Owner |
|---|---|---|
| `rps/data.py` | Loading, LSTM input windows (`build_windows`) | Selina (windows); Zunaer (cleaning and split, replacing the temporary `load_rounds` and `temporary_split`) |
| `rps/evaluate.py` | Accuracy, cross-entropy, per-round results, confusion matrix | Selina, shared |
| `rps/lstm.py` | LSTM model, training with early stopping | Selina |
| `rps/game.py` | `Agent` interface, expected-reward move choice, game loop | Selina |
| `rps/agents.py` | Random, static (LSTM) and blended agents | Selina |
| `rps/baselines.py` | Frequency and Markov predictors (to add) | Zunaer |
| `rps/opponents.py` | Simulated opponents and online-only agent (to add) | Zunaer |
| `scripts/run_lstm_experiments.py` | E1, E2, E3 -> `results/` | Selina |
| `scripts/run_agent_games.py` | Agents vs opponents -> `results/` | Selina |

## Conventions (please follow so the pieces fit)

* Moves are ints: 0 = rock, 1 = paper, 2 = scissors. Outcomes: 0 = win, 1 = tie, 2 = loss,
  from the human's point of view. -1 marks a timeout.
* Every predictor returns an (N, 3) probability array on the rows of `meta` from
  `data.build_windows(...)`, so all models are scored on the same rounds (rounds 11 to 300,
  skipping windows that touch a timeout). Score with `rps.evaluate`.
* Every agent subclasses `rps.game.Agent` and implements `predict(history)` (probabilities
  of the opponent's next move), optionally `act(history)`, `update(my_move, opp_move)` and
  `reset()`. `history` is a list of `(my_move, opp_move)` for rounds already played.
  `play_game` makes both players choose before either sees the other's move.
* Choose settings on the validation split only; the test split is scored once.

## AI use

Parts of this code were drafted with AI assistance (Claude) and reviewed by the team.
