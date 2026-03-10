# ATP Match Outcome Prediction — Research Strategy

## Problem Statement

Binary classification: given two ATP players and pre-match context, predict the match winner. Evaluation includes both accuracy and probabilistic calibration (log loss).

---

## Dataset

`tennis_data/derived_data/complete.csv` — ~30,745 ATP tour-level matches, 2015–2026, 49 columns.

**Time-based split:**

| Split | Period |
|-------|--------|
| Train | 2015–2023 |
| Validation | 2024 |
| Test | 2025–2026 |

---

## Features

All features are computed from information available before the match starts (no leakage from match stats).

**Ranking & biographical**
- ATP ranking and ranking points for each player
- Ranking differential
- Age, height, handedness

**Surface**
- Surface type (Hard, Clay, Grass)
- Per-player surface win rate (rolling)

**Recent form**
- Rolling win rate over last N matches
- Rolling win rate on current surface

**Head-to-head**
- Overall H2H record
- Surface-specific H2H record

**Match context**
- Tournament level (Grand Slam, Masters, 500/250)
- Round
- Best-of (3 vs 5)

---

## Baseline Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline; interpretable |
| Random Forest | Non-linear; robust to scale |
| XGBoost | Strong tabular baseline |
| LightGBM | Fast; handles missing values natively |

Hyperparameters tuned via `TimeSeriesSplit` cross-validation to respect temporal ordering.

---

## Evaluation

| Metric | Description |
|--------|-------------|
| Accuracy | Overall match prediction rate |
| Log Loss | Calibration of predicted probabilities |
| AUC-ROC | Discrimination ability |

Literature benchmarks for ATP match prediction: ~65–70% accuracy, log loss < 0.55.

---

## References

- Kovalchik, S. (2016). Searching for the GOAT of tennis win prediction. *Journal of Quantitative Analysis in Sports*, 12(3).
- Sipko, M., & Knottenbelt, W. (2015). Machine learning for the prediction of professional tennis matches. *Imperial College London*.
- Sackmann, J. Tennis Abstract. [tennisabstract.com](https://tennisabstract.com/)
