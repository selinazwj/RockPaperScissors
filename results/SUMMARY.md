# Results so far (2026-11-25, Selina's branch)

Setup: temporary cleaning and split (217 games; 153 train / 34 val / 30 test players),
all models scored on the same 8,437 test rounds (rounds 11-300, timeouts skipped).
LSTM: 1 layer, hidden 32, Adam lr 0.001, batch 256, early stopping (patience 5) on
validation cross-entropy, 3 seeds. Rerun after Zunaer's final cleaning and split.

## E1 and E2 (validation, mean of 3 seeds)

| Experiment | k | Features | Val. accuracy | Val. cross-entropy |
|---|---|---|---|---|
| E1 | 3 | C | 65.0% | 0.751 |
| E1 | 5 | C | 65.9% | 0.738 |
| E1 | 10 | C | 67.7% | 0.707 |
| E2 | 10 | A (human only) | 65.8% | 0.777 |
| E2 | 10 | B (+ bot moves) | 67.4% | 0.711 |
| E2 | 10 | C (+ outcome) | 67.7% | 0.707 |

Selected: k = 10, feature set C.

## E3 (test)

| Model | Accuracy | Cross-entropy |
|---|---|---|
| Random (expected) | 33.3% | 1.099 |
| Global frequency | 35.0% | 1.097 |
| First-order Markov | 42.1% | 1.074 |
| Outcome-aware Markov | 42.5% | 1.056 |
| Count model, last 2 rounds of both moves | 63.9% | 0.811 |
| Count model, last 3 rounds of both moves | 65.1% | 0.749 |
| LSTM (k = 10, C), 3 seeds | 69.1% +/- 0.2 | 0.678 |

The baseline rows are quick reference numbers computed by Selina; Zunaer's
rps/baselines.py produces the official ones.

## First look at E4 (LSTM seed 0, test)

By bot strategy (4-5 test players each, so treat with caution): from 57%
(win_nil_lose_positive) to 87% (opponent_prev_move_nil).
By previous outcome: 80% after a win, 58% after a tie, 54% after a loss.

## Hyperparameter variants (validation, k = 10, feature set C, mean of 3 seeds)

| Version | Change from v1 | Val. accuracy | Val. cross-entropy | Epochs |
|---|---|---|---|---|
| v1 | (1 layer, hidden 32, lr 0.001, no dropout) | 67.7% | 0.707 | 37 |
| v2 | hidden 64 | 67.6% | 0.704 | 26 |
| v3 | learning rate 0.003 | 67.8% | 0.704 | 17 |
| v4 | 2 layers + dropout 0.2 | 67.6% | 0.702 | 35 |

No variant beats v1 by more than the spread between seeds (about 0.2 points of
accuracy), so we keep the simpler v1. Model capacity is not the bottleneck.
