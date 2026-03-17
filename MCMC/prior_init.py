"""
Stage 1A — Prior Initialization

Responsibilities:
  1. Fit the rank -> skill mapping mu(rank) = -b * log(rank) + C
     from historical match outcomes using MLE.
  2. Define PlayerState: per-player Gaussian approximation
     (mean 3-vector, Cholesky 3x3) over (clay, hard, grass) skills.
  3. Initialize all player states from their first observed ranking.
  4. Compute dynamic process noise tau_i^2(t) = tau_base^2 / (1 + lam * n_i(t)).

Outputs:
  tennis_data/derived_data/player_states_init.pkl
  tennis_data/derived_data/rank_skill_map.pkl

Usage:
  python -m MCMC.prior_init          # run standalone to build and save init states
  from MCMC.prior_init import ...    # import into hmc_match.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

# ── Constants ─────────────────────────────────────────────────────────────────

SURFACES    = ['clay', 'hard', 'grass']
SURF_IDX    = {s: i for i, s in enumerate(SURFACES)}   # clay=0, hard=1, grass=2

DATA_DIR    = Path('tennis_data/derived_data')
STATES_PATH = DATA_DIR / 'player_states_init.pkl'
MAP_PATH    = DATA_DIR / 'rank_skill_map.pkl'

# Default hyperparameters (tuned via cross-validation in Phase 3.3)
TAU_BASE_DEFAULT = 0.30   # base process noise SD
LAM_DEFAULT      = 0.05   # career stabilization rate
SIGMA0_DEFAULT   = 1.00   # initial per-surface skill SD (diagonal covariance)
RANK_CENTER      = 50     # rank at which mu(rank) = 0 (50th-ranked player as origin)


# ── Player state ──────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """
    Gaussian approximation of player skill posterior over (clay, hard, grass).

    The full covariance is stored as its lower-triangular Cholesky factor L,
    so cov = L @ L.T.  This is the form PyMC expects for non-centered
    parameterization:  theta = mean + L @ z,  z ~ N(0, I_3).
    """
    name:       str
    mean:       np.ndarray   # shape (3,) — skills (clay, hard, grass)
    chol:       np.ndarray   # shape (3,3) — lower-triangular Cholesky of cov
    n_matches:  int   = 0    # career match count at this point in time
    init_rank:  float = 500.0

    @property
    def cov(self) -> np.ndarray:
        return self.chol @ self.chol.T

    def with_process_noise(self, tau_base: float = TAU_BASE_DEFAULT,
                           lam: float = LAM_DEFAULT) -> 'PlayerState':
        """
        Return a new PlayerState whose covariance includes the drift noise
        for the upcoming match.

            tau^2(t) = tau_base^2 / (1 + lam * n_i(t))
            Sigma_prior = Sigma_posterior + tau^2(t) * I_3
        """
        tau_sq = process_noise(self.n_matches, tau_base, lam)
        new_cov = self.cov + tau_sq * np.eye(3)
        # Add tiny jitter to guarantee positive-definite
        new_cov += 1e-7 * np.eye(3)
        new_chol = np.linalg.cholesky(new_cov)
        return PlayerState(
            name      = self.name,
            mean      = self.mean.copy(),
            chol      = new_chol,
            n_matches = self.n_matches,
            init_rank = self.init_rank,
        )

    def surface_mean(self, surface: str) -> float:
        return float(self.mean[SURF_IDX[surface]])

    def surface_std(self, surface: str) -> float:
        idx = SURF_IDX[surface]
        return float(np.sqrt(self.cov[idx, idx]))


# ── Process noise ─────────────────────────────────────────────────────────────

def process_noise(n_matches: int, tau_base: float = TAU_BASE_DEFAULT,
                  lam: float = LAM_DEFAULT) -> float:
    """
    tau_i^2(t) = tau_base^2 / (1 + lam * n_i(t))

    Early career (n small): large tau -> model accepts rapid revisions.
    Established (n large):  small tau -> strong evidence required to shift estimate.
    """
    return tau_base**2 / (1.0 + lam * max(n_matches, 0))


# ── Rank -> skill mapping ─────────────────────────────────────────────────────

def fit_rank_skill_map(df: pd.DataFrame) -> tuple[float, float]:
    """
    Fit b in:  P(A beats B) = sigmoid( b * log(rank_B / rank_A) )

    This is MLE over all matches where both ranks are real (not imputed).
    Winner is always the row's winner, so outcome = 1 always.

    The rank->skill function is then:
        mu(rank) = -b * log(rank) + C,   C = b * log(RANK_CENTER)
    so that mu(RANK_CENTER) = 0.

    Returns
    -------
    b : float   — slope of log-rank skill map
    C : float   — intercept so mu(RANK_CENTER) = 0
    """
    # Filter to matches with real (non-imputed) ranks
    rank_imp = df['rank_imputed']
    if rank_imp.dtype == object:
        mask_imp = rank_imp.str.lower() == 'true'
    else:
        mask_imp = rank_imp.astype(bool)

    valid = df[
        ~mask_imp &
        df['winner_rank'].notna() &
        df['loser_rank'].notna()
    ].copy()

    print(f"  Rank map fit: {len(valid):,} matches with non-imputed ranks")

    # log(rank_B / rank_A) — positive when A is better-ranked (lower rank number)
    log_rank_diff = (
        np.log(valid['loser_rank'].values.astype(float)) -
        np.log(valid['winner_rank'].values.astype(float))
    )

    def neg_log_likelihood(params):
        b = params[0]
        # P(winner wins) = sigmoid(b * log_rank_diff)
        lp   = b * log_rank_diff
        # Numerically stable log-sigmoid
        ll   = np.where(lp >= 0,
                        -np.log1p(np.exp(-lp)),
                        lp - np.log1p(np.exp(lp)))
        return -ll.sum()

    result = minimize(neg_log_likelihood, x0=[0.8], method='L-BFGS-B',
                      bounds=[(0.01, 10.0)])
    b = float(result.x[0])
    C = b * np.log(RANK_CENTER)

    # Evaluate fit quality
    pred = sigmoid(b * log_rank_diff)
    brier = float(((pred - 1.0) ** 2).mean())  # outcome always 1

    print(f"  Fitted b = {b:.4f}   (Brier on training = {brier:.4f})")
    for rank in [1, 10, 50, 100, 200, 500, 1000]:
        mu_r = -b * np.log(rank) + C
        print(f"    mu(rank {rank:4d}) = {mu_r:+.3f}")

    return b, C


def rank_to_skill(rank: float, b: float, C: float) -> float:
    """Map ATP ranking to prior mean skill (log-linear model)."""
    return -b * np.log(max(float(rank), 1.0)) + C


# ── Player state initialization ───────────────────────────────────────────────

def build_initial_player_states(
    df:    pd.DataFrame,
    b:     float,
    C:     float,
    sigma0: float = SIGMA0_DEFAULT,
) -> dict[str, PlayerState]:
    """
    Create initial PlayerState for every player seen in preprocessed.csv.

    Prior mean: mu(entry_rank) applied equally across all three surfaces.
    Initial covariance: sigma0^2 * I_3 (independent surfaces, diagonal).

    The cross-surface correlation Sigma will be refined by the global HMC fit
    (Phase 3.1 global model).  Until then, diagonal initialization is used.

    Players whose rank was never observed get rank=500 (median tour).
    """
    print(f"\n  Building initial player states ...")
    print(f"  Covariance: sigma0={sigma0} * I_3 (diagonal, independent surfaces)")

    # Collect first observed rank for each player (scan chronologically)
    first_rank: dict[str, float] = {}
    for _, row in df.iterrows():
        w, l = row['winner_name'], row['loser_name']
        wr   = float(row['winner_rank']) if pd.notna(row['winner_rank']) else None
        lr   = float(row['loser_rank'])  if pd.notna(row['loser_rank'])  else None
        if w not in first_rank and wr is not None:
            first_rank[w] = wr
        if l not in first_rank and lr is not None:
            first_rank[l] = lr

    # Players with no rank observation → assign median (rank 500)
    all_players = set(df['winner_name']) | set(df['loser_name'])
    for p in all_players:
        if p not in first_rank:
            first_rank[p] = 500.0

    # Build states
    chol_init = np.eye(3, dtype=float) * sigma0
    states: dict[str, PlayerState] = {}
    for name, rank in first_rank.items():
        mu = rank_to_skill(rank, b, C)
        states[name] = PlayerState(
            name      = name,
            mean      = np.array([mu, mu, mu], dtype=float),
            chol      = chol_init.copy(),
            n_matches = 0,
            init_rank = rank,
        )

    print(f"  Initialized {len(states):,} player states")
    # Skill distribution summary
    skills = [rank_to_skill(r, b, C) for r in first_rank.values()]
    print(f"  Skill range: [{min(skills):.2f}, {max(skills):.2f}]   "
          f"mean = {np.mean(skills):.2f}")
    return states


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_player_states(states: dict[str, PlayerState], path: Path = STATES_PATH) -> None:
    # Serialize to plain dicts to avoid pickle __main__ class name issues
    serialized = {
        name: {
            'name':      ps.name,
            'mean':      ps.mean.tolist(),
            'chol':      ps.chol.tolist(),
            'n_matches': ps.n_matches,
            'init_rank': ps.init_rank,
        }
        for name, ps in states.items()
    }
    with open(path, 'wb') as f:
        pickle.dump(serialized, f)
    print(f"  Saved {len(states):,} player states -> {path}")


def load_player_states(path: Path = STATES_PATH) -> dict[str, PlayerState]:
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    return {
        name: PlayerState(
            name      = d['name'],
            mean      = np.array(d['mean'], dtype=float),
            chol      = np.array(d['chol'], dtype=float),
            n_matches = d['n_matches'],
            init_rank = d['init_rank'],
        )
        for name, d in raw.items()
    }


def save_rank_map(b: float, C: float, path: Path = MAP_PATH) -> None:
    with open(path, 'wb') as f:
        pickle.dump({'b': b, 'C': C}, f)
    print(f"  Saved rank skill map (b={b:.4f}, C={C:.4f}) -> {path}")


def load_rank_map(path: Path = MAP_PATH) -> tuple[float, float]:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['b'], d['C']


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STAGE 1A — PRIOR INITIALIZATION")
    print("=" * 60)

    print("\nLoading preprocessed.csv ...")
    df = pd.read_csv(DATA_DIR / 'preprocessed.csv', low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  {len(df):,} matches, {df['winner_name'].nunique()} unique winners")

    print("\nStep 3.2 -- Fitting rank -> skill mapping ...")
    b, C = fit_rank_skill_map(df)
    save_rank_map(b, C)

    print("\nStep 3.1 -- Initializing player states ...")
    states = build_initial_player_states(df, b, C)
    save_player_states(states)

    print("\n" + "=" * 60)
    print("STAGE 1A COMPLETE")
    print(f"  player_states_init.pkl  -> {len(states):,} players")
    print(f"  rank_skill_map.pkl      -> b={b:.4f}, C={C:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
