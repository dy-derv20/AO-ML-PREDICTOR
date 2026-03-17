# Hierarchical Bayesian Latent Skill Model with Tournament-Structured Contextual Priors and Sequential Within-Tournament Posterior Updating for Calibrated Upset Prediction

> **Research Note:** This document is a living specification. Sections marked with a warning indicate design choices that remain open, are potentially sensitive to implementation details, or where the literature does not provide clear precedent. These are flagged not as weaknesses but as honest research questions to be resolved empirically.

---

## Implementation Status

| Stage | Status | Output |
|---|---|---|
| Stage 0 — Data Construction | **Complete** | `preprocessed.csv`, `upset_rate_matrix.csv`, `pressure_coefficients.csv` |
| Stage 1A — Prior Initialization | **Partial** — rank map + player states done; global Σ fit pending | `player_states_init.pkl`, `rank_skill_map.pkl` |
| Stage 1B — Sequential Update Loop | **Prototype validated** (Phase 4.1) — parallel execution pending | `MCMC/hmc_match.py`, `stage1b_prototype.py`, `prototype_cache.pkl` |
| Stage 1C — Tournament Inference Mode | Not started | — |
| Stage 2A — Logistic Regression Baseline | Not started | — |
| Stage 2B — Normalizing Flow | Not started | — |
| Stage 3 — Evaluation | Not started | — |

**Key files (Stage 0 outputs):**
- `tennis_data/derived_data/preprocessed.csv` — 27,683 rows x 65 columns, sorted `[tourney_date, tourney_id, round_order, match_num]`
- `tennis_data/derived_data/upset_rate_matrix.csv` — 220 cells, grouped by `(surface, tourney_name, round)` for 500+/1000/GS/ATP Finals; 250s pooled
- `tennis_data/derived_data/pressure_coefficients.csv` — empirical pi(r, tier) with shrinkage diagnostics
- `tennis_data/derived_data/complete_clean.csv` — cleaned raw input

**Stage 1A outputs:**
- `tennis_data/derived_data/player_states_init.pkl` — 797 PlayerState objects (mean 3d, Cholesky 3x3, init_rank, n_matches=0)
- `tennis_data/derived_data/rank_skill_map.pkl` — fitted (b=0.5532, C=2.1642) for mu(rank) = -b*log(rank)+C

**Stage 1B outputs (prototype):**
- `tennis_data/derived_data/prototype_cache.pkl` — feature_rows, diagnostics, player_states from first N matches

**Entry points:**
- `preprocessing.py` — Stage 0 full pipeline
- `add_batch_columns.py` — fast fixup for existing preprocessed.csv
- `MCMC/prior_init.py` — Stage 1A: rank map + player state initialization
- `MCMC/hmc_match.py` — Stage 1B: single-match PyMC model + update logic
- `MCMC/stage1b_prototype.py` — Stage 1B prototype runner (Phase 4.1 validation)

---

## Overview

This system is a two-stage probabilistic architecture for predicting professional tennis match upsets. Stage 1 infers time-varying, surface-specific latent skill for every ATP player via a sequential Hierarchical Bayesian model fit with HMC/NUTS. Stage 2 consumes the full posterior predictive distribution produced by Stage 1 — not just scalar summaries — and learns to predict upset probability using either an interpretable logistic regression ablation or a normalizing flow that embeds the full distributional shape into a learned latent space. The system is extended with tournament-structural priors, within-tournament momentum tracking, and a live inference mode that updates round-by-round as a tournament progresses.

---

## Notation Reference

| Symbol | Meaning |
|---|---|
| $\theta_i^s$ | Latent skill of player $i$ on surface $s$ |
| $\mathcal{H}_{<t}$ | Full match history observed strictly before match $t$: $\{(A_k, B_k, y_k, s_k)\}_{k=1}^{t-1}$ |
| $s \in \mathcal{S}$ | Surface: $\mathcal{S} = \{\text{clay, hard, grass}\}$ |
| $r$ | Tournament round: $\{R128, R64, R32, R16, QF, SF, F, RR\}$ — RR denotes ATP Finals group stage |
| $\text{tier}$ | Tournament tier: $\{\text{Grand Slam, ATP Finals, 1000, 500, 250}\}$ |
| $n_i(t)$ | Career match count of player $i$ at time $t$ |
| $\tau_i^2(t)$ | Dynamic process noise for player $i$ at time $t$ |
| $\lambda$ | Career stabilization rate for process noise |
| $\beta$ | Within-tournament momentum decay rate |
| $\pi(r, \text{tier})$ | Empirical pressure coefficient at round $r$ and tier — estimated from data, not hand-coded |
| $\rho_{s, \text{tourney}, r}$ | Historical upset rate at surface $s$, tournament, round $r$ — per named tournament for 500+, pooled for 250s |
| $S$ | Number of HMC posterior samples drawn per match |
| $p_k$ | $k$-th posterior predictive sample of $P(A \text{ wins})$ |
| $M_i(r)$ | Within-tournament momentum score for player $i$ at round $r$ |
| $Q_i(r)$ | Opponent quality trajectory for player $i$ at round $r$ |
| $W_i^{(r)}$ | Pressure-weighted win signal for player $i$ at round $r$ |

---

## STAGE 0 — Data Construction

> **Status: Complete.** All outputs exist on disk. Do not re-run `preprocessing.py` unless the source data changes — the H2H loop takes significant time.

### 0.1 Sources

- **Primary:** Jeff Sackmann ATP match records, 2015–2024
- **Extension:** TML-Database for 2025–2026 coverage
- **Point-by-point data:** Planned for select tournaments (Wimbledon, US Open)

> **Dataset scope:** The pipeline validation run uses 2015–2026 (~27,683 matches after cleaning). Extension to 2000–2026 (~70,000 matches) is planned after the architecture is validated — this requires a full recompute from scratch as the sequential posteriors are not reusable across dataset boundaries.

### 0.2 Preprocessing Steps

```
1. Load all match records
2. Drop non-ATP individual tour events:
     Davis Cup (team format, non-standard conditions)
     Olympics (special format, different motivation)
     Laver Cup, ATP Cup, United Cup (team events)
     Next Gen Finals (youth event, different rules)
3. Standardize tourney_level to taxonomy:
     G -> Grand Slam, M -> 1000, F -> ATP Finals
     Sackmann 'A' -> 500 or 250 by tournament name
4. Parse and sort strictly by match date --
   no exceptions, no random shuffling at any stage
5. Assign surface label s in {clay, hard, grass}
6. Assign round label r in {R128,R64,R32,R16,QF,SF,F,RR}
7. Compute historical upset rate matrix rho_{s, tourney, r}:
     rho = count(upsets) / count(matches) at each cell
     Grouping:
       - 500 / 1000 / Grand Slam / ATP Finals:
         per (surface, tourney_name, round)
         [individual tournament resolution -- AO, Wimbledon,
          Roland Garros, etc. each get their own cells]
       - 250: pooled across all 250s per (surface, round)
         [too few matches per individual 250 for stable rates]
     Upset defined by ATP ranking (winner_rank > loser_rank)
     for this bootstrap approximation only -- model-implied
     definition takes over as the training target in Stage 2.
     Output: upset_rate_matrix.csv (220 cells)
8. For each player i, initialize career match
   counter n_i = 0, increment after each match
9. Construct H2H records per player pair:
     overall win/loss
     surface-specific win/loss
     pressure-context win/loss (SF/F or tier in {1000, ATP Finals, Grand Slam})
     recency-filtered win/loss (last 2 years)
   [All H2H counts recorded strictly before each match]
10. Compute empirical pressure coefficients pi(r, tier):
     Signal: how much does the upset rate at (r, tier)
     deviate from the global baseline p0?
     Hierarchical structure:
       pi_cell(r, tier)     = upset_rate(r, tier) / p0
       pi_fallback(r, tier) = sqrt(pi_r(r) x pi_t(tier))
                              [geometric mean of round-only
                               and tier-only marginal coefficients;
                               still parameterized by both but
                               estimated from more data]
       w = min(n(r,tier) / 30, 1.0)   [shrinkage weight]
       pi_raw = w x pi_cell + (1-w) x pi_fallback
       pi(r, tier) = pi_raw / min(all pi_raw)  [normalize to min=1]
     Output: pressure_coefficients.csv
     [Empirically: Grand Slam SF = 1.0 (minimum -- least upset-prone
      after field is filtered); ATP Finals F = highest but sparse]
```

### 0.3 Batch Grouping Columns

Two columns are added at the end of Stage 0 to support Stage 1B's parallel execution structure:

**`round_order`** — integer encoding of round label:
```
R128=1, R64=2, R32=3, R16=4, QF=5, SF=6, F=7, RR=3
```
Enables within-tournament round sequencing without string comparison.

**`week_id`** — ISO calendar week string derived from `tourney_date`:
```
week_id = tourney_date.strftime('%G-W%V')   # e.g. "2023-W03"
```
Groups all tournaments starting in the same ISO calendar week — supports Level 1 parallelism (simultaneous tournaments fire together in Stage 1B).

**Important caveats:**
- `tourney_date` is the **tournament start date** for every match in that tournament, not a per-match play date. Every round of AO 2023 carries `tourney_date = 2023-01-16`.
- `match_num` ranges overlap heavily across same-week tournaments (e.g. Adelaide 1/2, Auckland, Pune all use match_nums 270-300), so match_num alone cannot distinguish them.
- The Stage 1B batch unit is `(tourney_id, round)`, not position in the sorted dataframe.

**Correct global sort key:**
```
[tourney_date, tourney_id, round_order, match_num]
```
This enforces: cross-week ordering by date -> within-week grouping by tournament -> within-tournament ordering by round -> within-round ordering by match.

### 0.4 Chronological Integrity

The entire dataset is treated as a single ordered stream. Train/test split is performed by time — the test set is a held-out final block of tournaments, never sampled randomly. This enforces strict causal ordering and prevents any form of data leakage.

> **Open question:** How large should the held-out test set be? One full season is a reasonable starting point, but the right balance between test set size and training data richness is worth empirical investigation. Similarly, whether to hold out specific tournaments vs a trailing time window is a design choice.

---

## STAGE 1A — Hyperparameter and Prior Initialization

### 1.1 Hierarchical Prior Structure

The prior over player skill is structured hierarchically:

$$\mu_s \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad \text{for each surface } s$$

$$\sigma_s \sim \text{HalfNormal}(\eta_s) \quad \text{for each surface } s$$

$$\theta_i^s \sim \mathcal{N}(\mu_s, \sigma_s^2) \quad \text{for each player } i, \text{ surface } s$$

Tournament-tier random effects are layered on top:

$$\gamma_{\text{tier}} \sim \mathcal{N}(0, \sigma_{\text{tier}}^2)$$

The effective prior mean for a new player entering at a given tier and surface is:

$$\theta_i^{s, \text{tier}} \sim \mathcal{N}(\mu_s + \gamma_{\text{tier}}, \sigma_s^2)$$

> **Design decision:** Surface-specific skills are modeled as correlated via a multivariate normal prior rather than as independent. Clay and hard court skills share substantial covariance (baseline players dominate both), while grass is more distinct. This structure is more realistic and avoids wasting information — a player with rich hard court history but sparse grass court history will have their grass skill shrunk toward the correlated population structure rather than toward a flat uninformative prior. The full prior becomes:
>
> $$\boldsymbol{\theta}_i = (\theta_i^{\text{clay}}, \theta_i^{\text{hard}}, \theta_i^{\text{grass}}) \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$
>
> Where $\boldsymbol{\Sigma}$ is a learned $3 \times 3$ covariance matrix (parameterized via a Cholesky factor for HMC compatibility) that encodes the cross-surface correlation structure from the full historical dataset.

### 1.2 Initialization for New Players

New players are not initialized with a flat uninformative prior. Their prior is conditioned on ATP entry ranking:

$$\theta_i^{s,(0)} \sim \mathcal{N}(\mu(\text{entry rank}), \sigma_0^2)$$

Where $\mu(\text{rank}) = -b \cdot \log(\text{rank}) + C$ is a log-linear skill map fitted by MLE on all non-imputed ranked matches:

$$P(\text{A beats B}) = \sigma\!\left(b \cdot \log\!\frac{\text{rank}_B}{\text{rank}_A}\right) \quad \Rightarrow \quad b = 0.5532, \quad C = b \cdot \log(50) = 2.164$$

The constant $C$ is chosen so $\mu(50) = 0$ (rank-50 player is the skill origin). Selected values: rank 1 → +2.16, rank 50 → 0.00, rank 200 → -0.77, rank 1000 → -1.66. Initial covariance is $\sigma_0^2 \mathbf{I}_3$ with $\sigma_0 = 1.0$ (independent surfaces), to be updated by the global Σ fit.

**The key property of hierarchical shrinkage:** Every retired historical player contributed to estimating $\mu_s$, $\sigma_s$, $\gamma_{\text{tier}}$, and the population-level drift parameters. These hyperparameters encode 20+ years of knowledge about what skill distributions look like at each tier and surface. A new player's prior inherits this structure immediately — without ever having played the historical players directly.

### 1.3 Global Hyperparameters Learned From Full History

| Parameter | Meaning | How Learned |
|---|---|---|
| $\mu_s, \sigma_s$ | Surface-specific skill population mean and spread | HMC over full dataset |
| $\boldsymbol{\Sigma}$ | $3 \times 3$ cross-surface skill covariance (Cholesky-parameterized) | HMC over full dataset |
| $\tau^2_{\text{base}}$ | Base process noise (skill drift rate) | HMC over full dataset |
| $\lambda$ | Career stabilization rate | HMC or cross-validated |
| $\beta$ | Within-tournament momentum decay | Cross-validated |
| $\pi(r, \text{tier})$ | Pressure coefficients per round/tier | Precomputed empirically from upset rate data with hierarchical shrinkage (see Stage 0 step 10). Used as a fixed feature in Stage 1B/2 — not re-estimated during HMC. |

> **Open question:** Whether $\lambda$ and $\beta$ are better treated as fully Bayesian hyperparameters inferred via HMC or tuned via cross-validation is uncertain. Full Bayesian treatment is principled but computationally expensive. A pragmatic approach is to fix them via cross-validation and treat sensitivity as a robustness check.

---

## STAGE 1B — Sequential Posterior Update Loop

This is the core of Stage 1. It processes every match in the dataset in strict chronological order. The batch unit is `(tourney_id, round)` — all matches within the same tournament round can be processed in parallel once their input posteriors are ready, but rounds must be processed sequentially within a tournament (round $r+1$ requires the updated posteriors from round $r$).

### 1.4 Dynamic Process Noise

Between matches, each player's latent skill is allowed to drift. The rate of allowed drift decreases as career match count grows:

$$\tau_i^2(t) = \frac{\tau_{\text{base}}^2}{1 + \lambda \cdot n_i(t)}$$

The random walk transition is:

$$\theta_i^{s,(t)} = \theta_i^{s,(t-1)} + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \tau_i^2(t))$$

Early career: $n_i(t)$ is small, $\tau_i^2$ is large — the model accepts rapid revision of skill estimates. Established players: $\tau_i^2$ has stabilized — the model requires substantial evidence to shift their estimate. This is the sole mechanism for temporal recency — no separate recency decay parameter is used.

### 1.5 Surface-Specific Bradley-Terry Likelihood

Given two players A and B competing on surface $s$, the probability that A wins is:

$$P(A \text{ beats } B \mid s) = \sigma(\theta_A^s - \theta_B^s) = \frac{1}{1 + \exp(-(\theta_A^s - \theta_B^s))}$$

This is the Bradley-Terry model applied surface-specifically. Each player maintains a separate latent skill per surface, each with its own hierarchical prior. The surface skills are allowed to diverge substantially — a player may have high clay skill and low grass skill simultaneously.

> **Relation to Glicko-2:** Glicko-2 is conceptually a closed-form Bayesian approximation of a similar pairwise comparison model with temporal drift. This system is more general: it uses HMC for exact posterior inference rather than closed-form approximations, and it nests the Bradley-Terry likelihood inside a full hierarchical structure. The two are not competing — Glicko-2 inspired the design, but this system supersedes it in expressiveness.

### 1.6 HMC/NUTS Posterior Sampling

For each match $t$, after forming the joint prior $P(\theta_A^s, \theta_B^s \mid \mathcal{H}_{<t})$:

```
1. Apply process noise to both player priors:
     Sigma_prior = Sigma_posterior + tau_i^2(t) * I_3
2. Build 6-parameter PyMC model (z_A, z_B in R^3, non-centered)
3. Run NUTS with cores=1 (sequential chains avoids multiprocess overhead):
     draws=200, tune=200, chains=2 (prototype)
     draws=400, tune=400, chains=4 (production -- reduces R-hat noise)
4. Compute posterior predictive samples:
     p_k = sigma(theta_A[s](k) - theta_B[s](k))  for k = 1,...,S
5. Diagnostics: check R-hat < 1.01, ESS > 400
   [flag matches where sampling was problematic]
6. Store warm-start values (posterior means in z-space) for next match
```

**Phase 4.1 prototype validated:**
- Non-centered parameterization: 0 divergences across all test matches
- Posterior direction: 100% correct (winner skill up, loser down on match surface)
- Off-surface skills (not constrained by current match) remain close to prior — correct
- `var_epist ≈ 0` at dataset start is expected: prior-dominated early in timeline, aleatoric uncertainty dominates
- Performance: ~4s/match without C++ compiler; ~0.3s expected with `conda install gxx`
- R-hat with 2 chains is borderline (expected); 4 chains recommended for production

> **Performance note:** PyTensor requires a C++ compiler for optimal speed. Without it, Python-mode execution degrades performance ~10×. Run `conda install gxx` to enable compiled backends before the full Phase 4.3 pipeline run.

> **Open question:** The choice of $S$ (number of posterior samples) involves a tradeoff between computational cost and quality of distributional characterization for the flow model. A minimum of 500-1000 samples is likely needed for the flow to see meaningful distributional shape. The right value should be determined empirically by checking when flow performance stabilizes as $S$ increases.

### 1.7 Pre-Match Feature Extraction

Before observing match outcome $y_t$, record the following feature vector:

**Uncertainty statistics from $\{p_1, \ldots, p_S\}$:**

$$\mu_{\text{pred}} = \frac{1}{S}\sum_k p_k$$

$$\sigma^2_{\text{total}} = \frac{1}{S}\sum_k (p_k - \mu_{\text{pred}})^2$$

$$\sigma^2_{\text{aleat}} = \frac{1}{S}\sum_k p_k(1 - p_k)$$

$$\sigma^2_{\text{epist}} = \sigma^2_{\text{total}} - \sigma^2_{\text{aleat}}$$

$$\text{skewness} = \frac{1}{S}\sum_k \left(\frac{p_k - \mu_{\text{pred}}}{\sigma_{\text{total}}}\right)^3$$

$$\text{tail\_mass} = \frac{1}{S}\sum_k \mathbf{1}[p_k < 0.3]$$

**Raw sample array:** $\{p_1, \ldots, p_S\}$ stored for the normalizing flow.

**Tournament context features** (all precomputed in Stage 0, available as columns in `preprocessed.csv`):

```
surface s
tier
round r
round_order          [integer round encoding for sequencing]
week_id              [ISO calendar week for parallelism grouping]
rho_{s, tourney, r} [upset_rate column -- per named tournament for
                     500+, pooled ATP 250 for 250-level events]
pi(r, tier)          [pi column -- empirical pressure coefficient]
```

**Within-tournament state features** (defined in Stage 1C):

```
M_A(r) - M_B(r)      [momentum differential]
Q_A(r) - Q_B(r)      [opponent quality differential]
W_A^(r), W_B^(r)     [pressure-weighted win signals]
H2H: overall, surface-specific,
     pressure-context, recency-filtered
surprise_A, surprise_B
```

### 1.8 Post-Match Update

After observing $y_t$:

```
1. Update posterior via standard (unweighted) likelihood
   [Temporal adaptation handled by process noise tau_i^2(t) alone]
2. Update within-tournament state:
     W_i^(r) = pi(r, tier) * sigma(theta_i^s - theta_opp^s)^{-1}
     M_i(r)  = sum_{k<=r} W_i^(k) * exp(-beta * (r - k))
     Q_i(r)  = running mean of {mu_opp^s} for all
               opponents beaten to reach round r
3. Update surprise score:
     surprise_i += (1 - p_match)  [accumulated upset surprise]
4. Warm-start parameters stored for next match
5. Increment n_i(t) for both players
```

---

## STAGE 1C — Tournament Inference Mode

This mode activates at the start of any tournament for which predictions are being generated. It preserves all Stage 1B mechanics and adds tournament-level tracking.

### 1.9 Pre-Tournament Initialization

```
1. Load current posteriors P(theta_i^s | H_<tournament)
   for all players in the published draw
2. Construct full bracket tree from draw
3. Assign pi(r, tier) values from pressure_coefficients.csv
4. For each R1 matchup, compute pre-tournament:
     P(upset) using Stage 2 model with current features
5. Propagate uncertainty through bracket:
     P(player i reaches round r) for all i, r
   [computed by integrating over all possible
    paths through the bracket tree]
```

> **Open question:** Bracket propagation — computing $P(\text{player } i \text{ reaches round } r)$ exactly — requires summing over an exponentially large number of paths. In practice this is approximated, either by Monte Carlo simulation over the bracket or by dynamic programming with independence assumptions between matches. The right approximation strategy is an implementation detail to resolve.

### 1.10 Within-Tournament Round-by-Round Update

**Unit of update:** The completed round, not the individual match. Within a round, individual match posteriors update as results arrive. Within-tournament state features for the next round are only finalized once the full round is complete, because $M_i(r)$, $Q_i(r)$, and $\text{surprise}_i$ require knowing all opponents who advanced.

```
As each match within round r completes:
  -> Update posterior for both players
     via warm-started lightweight HMC
  -> Do NOT yet compute next-round features

After full round r completes:
  -> Finalize M_i(r), Q_i(r), surprise_i
     for all advancing players
  -> Recompute bracket survival probabilities
  -> Compute full feature vectors for
     all round r+1 matchups
  -> Generate P(upset) predictions for
     upcoming round
```

**Derived within-tournament quantities:**

$$Q_i(r) = \frac{1}{r-1} \sum_{k=1}^{r-1} \mu_{o_k}^s$$

Where $o_k$ is the opponent beaten in round $k$ and $\mu_{o_k}^s$ is their posterior mean skill on surface $s$ — a quantity already computed by Stage 1B.

$$W_i^{(r)} = \pi(r, \text{tier}) \cdot \left[\sigma(\theta_i^s - \theta_{o_r}^s)\right]^{-1}$$

$$M_i(r) = \sum_{k=1}^{r} W_i^{(k)} \cdot \exp(-\beta \cdot (r - k))$$

$$\text{surprise}_i = \frac{1}{r-1} \sum_{k=1}^{r-1} (1 - p_k)$$

Where $p_k$ is the pre-match $P(i \text{ wins})$ computed by Stage 2 before match $k$ — already produced by the pipeline.

**Additional live inference features:**

```
delta_mu_A = mu_A^s(current) - mu_A^s(pre-tournament)
delta_mu_B = mu_B^s(current) - mu_B^s(pre-tournament)
rounds_observed = number of completed rounds
```

---

## STAGE 2A — Ablation Baseline: Logistic Regression

### 2.1 Purpose

This model serves two roles: it is the interpretable baseline that establishes which uncertainty features are individually predictive of upsets, and it is the ablation that Stage 2B must outperform to justify the additional complexity of the normalizing flow.

### 2.2 Full Feature Vector

```
Uncertainty (from Stage 1B):
  mu_pred, sigma^2_epist, sigma^2_aleat, skewness, tail_mass

Tournament context:
  surface s (encoded), tier (encoded), round r (encoded)
  rho_{s, tourney, r}  [upset_rate -- per named tournament
                        for 500+, pooled for 250s]
  pi(r, tier)          [empirical pressure coefficient]

Within-tournament state:
  M_A(r) - M_B(r)
  Q_A(r) - Q_B(r)
  W_A^(r), W_B^(r)
  H2H: overall, surface, pressure-context, recency
  surprise_A, surprise_B

Live mode additions (if active):
  delta_mu_A, delta_mu_B
  rounds_observed
```

### 2.3 Training

```
Target: upset in {0,1}
        [1 if the player with lower model-implied
         win probability (mu_pred < 0.5) won]
Model:  Logistic regression with L2 regularization
Split:  Strict chronological -- final N tournaments
        held out, no random sampling
```

> **Design decision:** Upset is defined by **model-implied probability**, not ATP ranking. A match is an upset if the player the model assigned $\mu_{\text{pred}} < 0.5$ to won. This is internally consistent — the model's own skill estimates determine who was the favourite. The ranking-based upset flag (`is_upset_rank`) is retained in the dataset as a bootstrap approximation used only to compute $\rho_{s,\text{tourney},r}$ before Stage 1B runs.

Coefficients are read directly to identify which uncertainty statistics are most predictive of upsets.

---

## STAGE 2B — Novel Model: Normalizing Flow

### 2.4 Architecture Motivation

The logistic regression ablation discards the shape of the posterior predictive distribution — it only sees hand-crafted moments. Two matches can have identical mean and variance but fundamentally different distributional shapes:

```
Match 1: unimodal at 0.55
         -> mild favorite, low uncertainty

Match 2: bimodal with mass at 0.3 and 0.8
         -> two plausible worlds,
           one dominant for each player
```

These have similar moments but very different upset implications. The normalizing flow sees the full shape and learns what matters.

### 2.5 Flow Model

```
Input per match:
  Empirical distribution over [0,1]
  from S posterior predictive samples {p_1,...,p_S}

Flow learns invertible transformation:
  f: [distribution shape] -> latent vector z in R^d

Concatenation:
  (z, x_match) where x_match is the full
  context feature vector from Stage 2A

Classifier head:
  (z, x_match) -> P(upset)

Training:
  Flow + classifier trained jointly
  Model discovers which aspects of
  distributional shape predict upsets
```

> **Open question:** The specific flow architecture — Real NVP, Neural Spline Flow, or another variant — is an open choice. The input representation of the empirical distribution also requires a decision: histogram binning, kernel density estimate evaluated at fixed grid points, or sorted sample quantiles are all viable. The right choice likely depends on $S$ and should be ablated. This is one of the more technically uncertain parts of the pipeline.

> **Open question:** The dimensionality $d$ of the latent embedding $z$ is a hyperparameter. Too small and shape information is lost. Too large and the classifier overfits. Should be tuned via held-out validation.

### 2.6 What The Flow Learns

The flow does not have memory of specific tournaments. It learns the geometry of uncertainty signatures that co-occur with upsets across all training data. Tournament context features ($\rho_{s,\text{tourney},r}$, $\pi(r,\text{tier})$, $M_i(r)$, etc.) anchor the learned geometry to the correct region of the latent space. At inference time, a new tournament with a similar distributional signature to historically upset-prone tournaments will produce a similar latent embedding $z$ and thus a similar elevated $P(\text{upset})$.

---

## STAGE 3 — Evaluation

### 3.1 Standard Metrics

| Metric | What It Measures | Target |
|---|---|---|
| Accuracy | Raw prediction rate | Competitive with ~70% literature ceiling |
| Brier Score | Calibration quality | Lower than discriminative baselines |
| Log-loss | Proper scoring rule | Lower than discriminative baselines |

### 3.2 Upset-Specific Metrics

```
Upset-conditioned accuracy:
  Accuracy restricted to matches where
  true label is upset
  [standard models perform near chance here]

Epistemic uncertainty vs realized upset rate:
  Bin matches by sigma^2_epist quartile
  Compute empirical upset rate per bin
  Test: does higher sigma^2_epist predict
        higher realized upset rate?

Round-stratified performance:
  Does model improvement over baseline
  concentrate in late rounds where
  within-tournament state is richest?
```

### 3.3 Ablation Comparison

```
Flow model vs logistic regression:
  If flow significantly outperforms on
  upset-conditioned accuracy -> distributional
  shape carries information beyond moments
  -> Novel empirical finding

Latent space analysis:
  Do high-epistemic matches cluster separately
  from high-aleatoric in z-space?
  Do Grand Slam matches occupy different
  region than ATP 250?
```

### 3.4 Live Inference Evaluation

```
Does P(upset) improve round-by-round
as within-tournament evidence accumulates?

Compare R1 predictions vs QF predictions
for same eventual matchup --
does the model get sharper?
```

### 3.5 Betting Simulation

```
For each match in held-out test set:
  1. Compute model P(upset)
  2. Convert bookmaker odds to implied probability
  3. Compute edge: model P - implied P
  4. If edge > 0: apply Kelly Criterion sizing
       f* = edge / (1 - implied P)
  5. Simulate bankroll evolution over tournament season

Output: ROI curve, Sharpe ratio of returns,
        drawdown analysis

Note: Positive ROI on held-out data is the
strongest real-world validation of calibration
quality -- a miscalibrated model cannot
systematically beat bookmaker odds.
```

> **Important caveat:** Betting simulation results are sensitive to which bookmaker odds are used and the assumed market efficiency. Results should be interpreted as a calibration validation tool, not as a financial claim. Fractional Kelly (e.g., half-Kelly) is recommended for robustness.

---

## Full Information Flow

```
ATP History (2015-2026, ~27,683 matches)
[Extension to 2000-2026 planned after architecture validation]
          |
          v
STAGE 0: [COMPLETE]
         Drop non-ATP events -> standardize tier/round labels
         Chronological ordering, surface/tier/round labels
         Upset rate matrix rho_{s,tourney,r}:
           per named tournament for 500+/1000/GS/ATP Finals
           pooled "ATP 250 (pooled)" for 250-level events
         Career counters n_i(t), H2H records (4 variants)
         Empirical pressure coefficients pi(r,tier):
           upset rate deviation from baseline + shrinkage
           toward factored marginal fallback
         Batch grouping: round_order, week_id
         Sort: [tourney_date, tourney_id, round_order, match_num]
          |
          v
STAGE 1A: Population hyperparameters learned:
          mu_s, Sigma (Cholesky), tau^2_base, lambda, beta
          New player priors initialized from entry rank
          |
          v
STAGE 1B: For each (tourney_id, round) batch in time order:
          |-- Dynamic process noise applied tau_i^2(t)
          |-- HMC/NUTS -> S posterior samples
          |-- Posterior predictive {p_1,...,p_S} computed
          |-- Pre-match feature vector recorded
          |-- Match outcome observed
          |-- Standard likelihood update (no recency weighting --
          |   temporal adaptation via tau_i^2(t) only)
          +-- Within-tournament state updated
          |
          v
STAGE 1C: [Tournament inference mode]
          Bracket constructed -> round-by-round updates
          Live posterior updates as rounds complete
          |
          v
Full feature vector per match:
  (uncertainty statistics, raw samples,
   tournament context, within-tournament state,
   live features if active, true upset label)
          |
          v
     +----+----+
     |         |
     v         v
STAGE 2A    STAGE 2B
Logistic    Normalizing Flow
Regression  encodes {p_1,...,p_S} -> z
     |      concatenates (z, x_match)
     v      -> P(upset)
P(upset)
     +----+----+
          |
          v
STAGE 3: Evaluation
  Accuracy, Brier, Log-loss
  Upset-conditioned accuracy
  Epistemic uncertainty vs realized upset rate
  Round-stratified performance
  Ablation: flow vs logistic regression
  Betting simulation: Kelly ROI on held-out season
```

---

## Open Research Questions Summary

1. ~~Should surface-specific skills be modeled as independent or correlated (multivariate normal)?~~ **Resolved: correlated via multivariate normal with learned Cholesky covariance.**
2. What is the right $S$ (HMC sample count) for stable flow performance?
3. Should $\lambda$ and $\beta$ be fully Bayesian or cross-validated?
4. What is the best approximation strategy for bracket propagation probabilities?
5. Which normalizing flow architecture (Real NVP, Neural Spline, etc.) is most appropriate for this input structure?
6. What is the best representation of the empirical distribution as input to the flow (histogram, KDE, quantiles)?
7. ~~Should "upset" be defined by ATP ranking or by model-implied probability?~~ **Resolved: model-implied. Upset = player with mu_pred < 0.5 wins. Ranking-based flag retained only for bootstrapping rho_{s,tourney,r}.**
8. How large should the held-out test set be — one season, two seasons, specific tournaments?
9. ~~Recency-weighted likelihood (exponential decay on observations)?~~ **Resolved: not used. Temporal adaptation handled entirely by dynamic process noise tau_i^2(t) — no separate recency parameter.**
10. ~~Should pressure coefficients pi(r, tier) use a static structural formula?~~ **Resolved: empirical estimation from upset rate data with hierarchical shrinkage toward a factored marginal fallback. Precomputed in Stage 0 and used as a fixed feature.**
11. Does the within-tournament momentum signal ($M_i(r)$ differential) add predictive power beyond the skill gap alone?
