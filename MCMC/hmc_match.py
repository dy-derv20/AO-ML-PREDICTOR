"""
Stage 1B — Single-Match HMC Update

Core inference unit: one Bayesian update per match.

Model structure
---------------
Given current posteriors for player A and player B (each a 3-surface Gaussian),
the match on surface s is modeled as:

    theta_A ~ N(mu_A, Sigma_A)    [= prior after process noise]
    theta_B ~ N(mu_B, Sigma_B)

    P(A wins | surface s) = sigmoid(theta_A[s] - theta_B[s])   [Bradley-Terry]

    outcome = 1  (A is always the winner in our data)

Non-centered parameterization for HMC:
    z_A ~ N(0, I_3)  ->  theta_A = mu_A + L_A @ z_A   where L_A = chol(Sigma_A)
    z_B ~ N(0, I_3)  ->  theta_B = mu_B + L_B @ z_B

This yields 6 sampled parameters (z_A and z_B in R^3).

After sampling, the posterior over theta_A and theta_B is summarized
(mean + Cholesky) and stored as the updated PlayerState for the next match.

Usage
-----
    from MCMC.hmc_match import SingleMatchUpdater, compute_feature_stats
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import warnings
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from MCMC.prior_init import (
    PlayerState, SURF_IDX, process_noise,
    TAU_BASE_DEFAULT, LAM_DEFAULT,
)

warnings.filterwarnings('ignore', category=UserWarning)

# ── Default sampling parameters (Phase 4.1 prototype settings) ──────────────

DRAWS_DEFAULT  = 200    # posterior draws per chain
TUNE_DEFAULT   = 200    # warmup steps per chain
CHAINS_DEFAULT = 2      # 2 chains sufficient for within-match diagnostics
TARGET_ACCEPT  = 0.90


# ── PyMC model for a single match ─────────────────────────────────────────────

def build_match_model(
    state_A:  PlayerState,
    state_B:  PlayerState,
    surf_idx: int,
) -> pm.Model:
    """
    Build a PyMC model for inferring theta_A and theta_B from one match.

    Parameters
    ----------
    state_A : PlayerState
        Prior for player A **after** process noise has been applied
        (call state_A.with_process_noise() before passing here).
    state_B : PlayerState
        Prior for player B after process noise.
    surf_idx : int
        Surface index: clay=0, hard=1, grass=2.

    Returns
    -------
    pm.Model
        Compiled model (not yet sampled).
    """
    mu_A  = state_A.mean.astype(np.float64)    # (3,)
    L_A   = state_A.chol.astype(np.float64)    # (3,3) lower-triangular
    mu_B  = state_B.mean.astype(np.float64)
    L_B   = state_B.chol.astype(np.float64)

    with pm.Model() as model:
        # --- Player A: non-centered ---
        z_A     = pm.Normal('z_A', mu=0.0, sigma=1.0, shape=(3,))
        theta_A = pm.Deterministic('theta_A',
                      pt.as_tensor_variable(mu_A) + pt.dot(pt.as_tensor_variable(L_A), z_A))

        # --- Player B: non-centered ---
        z_B     = pm.Normal('z_B', mu=0.0, sigma=1.0, shape=(3,))
        theta_B = pm.Deterministic('theta_B',
                      pt.as_tensor_variable(mu_B) + pt.dot(pt.as_tensor_variable(L_B), z_B))

        # --- Bradley-Terry likelihood on observed surface ---
        skill_diff = theta_A[surf_idx] - theta_B[surf_idx]
        p_win = pm.Deterministic('p_win', pm.math.sigmoid(skill_diff))

        # Outcome is always 1: winner_name is always the winner
        _outcome = pm.Bernoulli('outcome', p=p_win, observed=np.array(1))

    return model


# ── Posterior extraction ──────────────────────────────────────────────────────

def extract_posterior_state(
    idata:    az.InferenceData,
    var_name: str,
    player:   PlayerState,
) -> PlayerState:
    """
    Summarize posterior samples for a player into a new PlayerState.

    Computes sample mean and sample covariance from the chains*draws samples
    of theta (shape: [chains, draws, 3]).  Adds a small jitter to guarantee
    positive-definite before computing the Cholesky factor.
    """
    samples = idata.posterior[var_name].values.reshape(-1, 3)   # (N, 3)
    mean    = samples.mean(axis=0)                               # (3,)
    cov     = np.cov(samples.T)                                  # (3,3)
    # Regularize: jitter + shrink toward prior to stabilize sparse matches
    cov    += 1e-6 * np.eye(3)
    chol    = np.linalg.cholesky(cov)

    return PlayerState(
        name      = player.name,
        mean      = mean,
        chol      = chol,
        n_matches = player.n_matches + 1,
        init_rank = player.init_rank,
    )


# ── Main update function ──────────────────────────────────────────────────────

class MatchResult:
    """Outputs of a single-match HMC update."""
    __slots__ = ('state_A', 'state_B', 'p_samples', 'idata',
                 'wall_time_s', 'rhat_ok', 'ess_ok', 'n_divergences')

    def __init__(self, state_A, state_B, p_samples, idata,
                 wall_time_s, rhat_ok, ess_ok, n_divergences):
        self.state_A       = state_A
        self.state_B       = state_B
        self.p_samples     = p_samples
        self.idata         = idata
        self.wall_time_s   = wall_time_s
        self.rhat_ok       = rhat_ok
        self.ess_ok        = ess_ok
        self.n_divergences = n_divergences


def run_match_update(
    state_A:      PlayerState,
    state_B:      PlayerState,
    surface:      str,
    draws:        int   = DRAWS_DEFAULT,
    tune:         int   = TUNE_DEFAULT,
    chains:       int   = CHAINS_DEFAULT,
    target_accept: float = TARGET_ACCEPT,
    tau_base:     float = TAU_BASE_DEFAULT,
    lam:          float = LAM_DEFAULT,
    random_seed:  int   = 42,
) -> MatchResult:
    """
    Run the Bayesian update for a single match.

    Steps
    -----
    1. Apply process noise to both player priors.
    2. Build the PyMC model.
    3. Sample with NUTS.
    4. Extract updated posteriors and p_win samples.
    5. Check diagnostics.

    Parameters
    ----------
    state_A, state_B : PlayerState
        Current posteriors (BEFORE process noise — added internally).
    surface : str
        Match surface: 'clay', 'hard', or 'grass'.

    Returns
    -------
    MatchResult
    """
    surf_idx = SURF_IDX[surface]
    t0       = time.perf_counter()

    # 1. Apply process noise
    prior_A = state_A.with_process_noise(tau_base, lam)
    prior_B = state_B.with_process_noise(tau_base, lam)

    # 2. Build model
    model = build_match_model(prior_A, prior_B, surf_idx)

    # 3. Sample
    # cores=1 avoids subprocess spawn overhead for this tiny 6-parameter model.
    # With g++ installed (conda install gxx), compilation reduces wall time
    # from ~15s to ~0.3s per match.  Numpyro backend (JAX) is an alternative.
    with model:
        idata = pm.sample(
            draws         = draws,
            tune          = tune,
            chains        = chains,
            cores         = 1,       # sequential chains — avoids subprocess overhead
            target_accept = target_accept,
            progressbar   = False,
            random_seed   = random_seed,
        )

    wall_time_s = time.perf_counter() - t0

    # 4. Extract updated states
    new_A = extract_posterior_state(idata, 'theta_A', state_A)
    new_B = extract_posterior_state(idata, 'theta_B', state_B)

    # p_win posterior predictive samples  (chains * draws,)
    p_samples = idata.posterior['p_win'].values.flatten().astype(np.float32)

    # 5. Diagnostics
    summary      = az.summary(idata, var_names=['z_A', 'z_B'])
    rhat_ok      = bool((summary['r_hat'] <= 1.01).all())
    ess_ok       = bool((summary['ess_bulk'] >= 400).all())
    n_divergences = int(idata.sample_stats.diverging.values.sum())

    return MatchResult(
        state_A       = new_A,
        state_B       = new_B,
        p_samples     = p_samples,
        idata         = idata,
        wall_time_s   = wall_time_s,
        rhat_ok       = rhat_ok,
        ess_ok        = ess_ok,
        n_divergences = n_divergences,
    )


# ── Feature statistics from p_samples ────────────────────────────────────────

def compute_feature_stats(p_samples: np.ndarray) -> dict:
    """
    Compute Stage 1B uncertainty statistics from posterior predictive samples.

    Matches Section 1.7 of the algorithm spec.

    Parameters
    ----------
    p_samples : np.ndarray
        Array of P(A wins) samples of shape (S,), values in [0, 1].

    Returns
    -------
    dict with keys:
        mu_pred, var_total, var_aleat, var_epist, skewness, tail_mass
    """
    p     = p_samples.astype(float)
    mu    = float(p.mean())
    var_t = float(p.var())
    var_a = float((p * (1.0 - p)).mean())         # aleatoric
    var_e = max(var_t - var_a, 0.0)               # epistemic (clamp at 0)

    skew = 0.0
    if var_t > 1e-12:
        skew = float(((p - mu) ** 3).mean() / var_t ** 1.5)

    tail = float((p < 0.3).mean())

    return {
        'mu_pred':   mu,
        'var_total': var_t,
        'var_aleat': var_a,
        'var_epist': var_e,
        'skewness':  skew,
        'tail_mass': tail,
    }
