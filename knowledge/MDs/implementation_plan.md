# Implementation Plan
## Hierarchical Bayesian Tennis Upset Predictor

> Upset definition: **model-implied probability** (not ATP ranking).
> Dataset: 2015–2026 for pipeline validation; extend to 2000–2026 for final research run.

---

## PHASE 1 — Data Cleaning  ✅ Complete

**1.1** Drop non-ATP individual tour events:
  - Davis Cup (`tourney_level == 'D'`) — team format, non-standard conditions
  - Olympics (`tourney_level == 'O'`) — special event, different format
  - Laver Cup, ATP Cup, United Cup — team events embedded in Sackmann 'A' level
  - Next Gen Finals — youth event, different player pool

**1.2** Standardize `tourney_level` taxonomy to spec values:
  - `'G'` → `'Grand Slam'`
  - `'M'` → `'1000'`
  - `'F'` → `'ATP Finals'`
  - `'A'` → `'500'` or `'250'` based on tournament name lookup
  - `'250'`, `'500'` (TML) → already correct, keep as-is

**1.3** Fix null surfaces:
  - After dropping Davis Cup all null surfaces should be resolved
  - Verify; impute from tournament name if any remain

**1.4** Fix null `loser_id`:
  - One match: Hong Kong 2026, Botic Van De Zandschulp → player_id `122298`

**1.5** Resolve null ranks via ATP rankings file lookup:
  - Join on `player_id` + nearest weekly ranking date before match date
  - Rankings coverage: 2010–2019 (`atp_rankings_10s`), 2020–2023 (`atp_rankings_20s`), 2024 (`atp_rankings_current`)
  - For 2025–2026 and any remaining nulls: assign placeholder rank `1500` + set `rank_imputed = True` flag column

**1.6** Chronological sort and save:
  - Sort by `tourney_date` ascending, then `match_num` ascending
  - Save to `tennis_data/derived_data/complete_clean.csv`
  - Print final stats: row count, year distribution, null counts, level distribution

---

## PHASE 2 — Stage 0: Preprocessing Pipeline  ✅ Complete

**2.1** ✅ Compute historical upset rate matrix `ρ_{s, tourney, r}`:
  - Grouping: per `(surface, tourney_name, round)` for 500/1000/Grand Slam/ATP Finals
  - 250s pooled as `ATP 250 (pooled)` per `(surface, round)` — too few matches per individual event
  - Upset defined by ATP ranking (bootstrap approximation) — model-implied definition takes over in Stage 2
  - Output: `upset_rate_matrix.csv` (220 cells)

**2.2** ✅ Initialize career match counters `n_i(t)` per player:
  - Process matches in strict chronological order
  - For each match, record `n_winner(t)` and `n_loser(t)` before incrementing

**2.3** ✅ Build H2H records per player pair:
  - Overall win/loss
  - Surface-specific win/loss
  - Pressure-context win/loss (SF/F or tier ∈ {1000, ATP Finals, Grand Slam})
  - Recency-filtered win/loss (last 2 years)
  - Updated after each match in chronological order

**2.4** ✅ Compute empirical pressure coefficients `π(r, tier)`:
  - Signal: upset rate at (r, tier) relative to global baseline p0
  - Hierarchical: cell-specific estimate shrunk toward geometric mean of round-only and tier-only marginals
  - Shrinkage weight: `min(n / 30, 1.0)` — sparse cells fall back to factored marginal
  - Normalized so minimum cell = 1.0
  - Output: `pressure_coefficients.csv`

**2.5** ✅ Save preprocessed dataset with all Stage 0 features appended:
  - Output: `preprocessed.csv` (27,683 rows × 65 columns)
  - Extra columns added beyond original plan: `round_order`, `week_id`
  - Sort key: `[tourney_date, tourney_id, round_order, match_num]`

---

## PHASE 3 — Stage 1A: Prior Initialization  ✅ Complete

**3.1** ✅ Define hierarchical prior with correlated surface skills:
  - `θ_i = (θ_i^clay, θ_i^hard, θ_i^grass) ~ N(μ, Σ)`
  - Initial covariance: `σ0² * I_3` (diagonal); cross-surface `Σ` to be learned in global fit
  - `PlayerState` dataclass: stores (mean: np.array[3], chol: np.array[3,3], n_matches, init_rank)
  - Implemented in `MCMC/prior_init.py`

**3.2** ✅ Build entry rank → prior mean mapping:
  - MLE fit: `P(A beats B) = σ(b * log(rank_B / rank_A))`; fitted `b = 0.5532`
  - Mapping: `μ(rank) = -b * log(rank) + C` where `C = b * log(50)` so `μ(50) = 0`
  - Interpretation: rank 1 → +2.16, rank 50 → 0.00, rank 200 → -0.77, rank 1000 → -1.66
  - Output: `tennis_data/derived_data/rank_skill_map.pkl`

**3.3** Set cross-validated hyperparameters: ← pending
  - `λ` (career stabilization), `β` (momentum decay) — defaults: `λ=0.05`, `τ_base=0.30`
  - Note: recency decay `α` is removed from the model — temporal adaptation handled entirely
    by dynamic process noise `τ_i²(t)`, no separate recency parameter
  - Cross-validate on a held-out slice before the main HMC run

**Outputs:**
  - `MCMC/prior_init.py` — `PlayerState`, rank map fit, player state initialization
  - `tennis_data/derived_data/player_states_init.pkl` — 797 player states initialized
  - `tennis_data/derived_data/rank_skill_map.pkl` — fitted (b, C)

---

## PHASE 4 — Stage 1B: Sequential HMC Loop

**4.1** ✅ Prototype single-match HMC update in PyMC:
  - Non-centered: `z_A, z_B ~ N(0, I_3)`, `θ = μ + L @ z`; 6 sampled parameters total
  - Likelihood: `Bernoulli(σ(θ_A[s] - θ_B[s])) = 1` (winner always correct)
  - Process noise applied to prior before inference: `Σ_prior = Σ_post + τ²(n) * I_3`
  - Validated: 0 divergences, 100% correct posterior direction (winner up, loser down)
  - Timing: ~4s/match without C++ compiler, ~0.3s expected with `g++` installed
    - **Note:** `conda install gxx` required for compiled PyTensor (5-10× speedup)
  - R-hat: borderline at 2 chains (expected); 4 chains for production run
  - `var_epist = 0` at dataset start is correct (prior-dominated; aleatoric dominates)
  - Implemented in `MCMC/hmc_match.py`, prototype runner `MCMC/stage1b_prototype.py`

**4.2** Implement parallel execution structure:
  - Level 1: simultaneous tournaments in same week → parallel workers
  - Level 2: matches within same round → parallel workers
  - Level 3: 2 chains per match (reduced from 4)
  - Barrier: full round must complete before within-tournament state is aggregated

**4.3** Run full pipeline over 2015–2026:
  - Estimated: 2–4 hours on 8 cores
  - Validate R-hat < 1.01, ESS > 400 for each match

**4.4** Cache all outputs:
  - Per-match: posterior predictive samples `{p_1,...,p_S}`, full feature vector, true outcome
  - Per-player: posterior mean and variance at each point in time
  - Cache is permanent — all Stage 2 work runs on top of this

---

## PHASE 5 — Stage 1C: Tournament Inference Mode

**5.1** Bracket construction from published draw

**5.2** Pre-tournament prediction for all R1 matchups

**5.3** Round-by-round posterior update as results arrive:
  - Finalize `M_i(r)`, `Q_i(r)`, `surprise_i`, `Δμ` after each full round
  - Recompute bracket survival probabilities
  - Generate `P(upset)` for next round matchups

---

## PHASE 6 — Stage 2A: Logistic Regression Baseline

**6.1** Assemble full feature vector from cached Stage 1B outputs

**6.2** Train logistic regression with L2 regularization on chronological split

**6.3** Evaluate: accuracy, Brier score, log-loss, upset-conditioned accuracy

**6.4** Read coefficients — identify which uncertainty statistics predict upsets

---

## PHASE 7 — Stage 2B: Normalizing Flow

**7.1** Choose architecture: Neural Spline Flow (start here)

**7.2** Input representation: sorted quantiles from `{p_1,...,p_S}` (50 quantile points)

**7.3** Implement: flow encoder → latent `z` → concatenate with `x_match` → classifier head

**7.4** Train jointly on GPU (Colab Pro+ or cloud); estimated 1–4 hours

**7.5** Ablation: flow vs logistic regression on upset-conditioned accuracy
  - If flow significantly outperforms → distributional shape carries information beyond moments

---

## PHASE 8 — Stage 3: Evaluation

**8.1** Standard metrics: accuracy, Brier, log-loss vs baselines

**8.2** Upset-specific: upset-conditioned accuracy, epistemic uncertainty bin analysis

**8.3** Round-stratified performance: does model improve in late rounds?

**8.4** Latent space analysis: do high-epistemic vs high-aleatoric matches cluster separately?

**8.5** Betting simulation: Kelly criterion on held-out season → ROI, Sharpe, drawdown

---

## DATA EXTENSION (after Phase 8 validates architecture)

- Obtain Sackmann 2000–2014 files, add to `match_by_match/`
- Rerun full pipeline from Phase 1 with 2000–2026 data
- Full recompute required — 2015–2026 cached outputs are invalidated
- Estimated: 4–6 hours on 16 cores
- Compare evaluation metrics: 2015–2026 baseline vs 2000–2026 full run
