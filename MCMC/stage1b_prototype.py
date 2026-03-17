"""
Stage 1B Prototype — Phase 4.1

Validates the single-match HMC update on the first N matches of the dataset.

Purpose
-------
- Confirm PyMC model builds and samples without errors.
- Verify posterior updates move in the correct direction (winner skill up,
  loser skill down relative to the prior).
- Measure per-match wall time to estimate full pipeline cost.
- Check that R-hat < 1.01 and ESS > 400 pass consistently.
- Profile divergences and flag any sampling issues.

Outputs
-------
  tennis_data/derived_data/prototype_cache.pkl
      dict with keys:
        'feature_rows'  : list of dicts (one per match, feature stats + metadata)
        'player_states' : dict[name -> PlayerState] at end of run
        'diagnostics'   : list of dicts (rhat_ok, ess_ok, n_div, wall_time_s)

Usage
-----
  python -m MCMC.stage1b_prototype             # runs first 50 matches
  python -m MCMC.stage1b_prototype --n 200     # first 200 matches
  python -m MCMC.stage1b_prototype --n 50 --draws 400 --chains 4
"""

import sys
import argparse
import pickle
import copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from MCMC.prior_init import (
    PlayerState, SURF_IDX, SURFACES,
    TAU_BASE_DEFAULT, LAM_DEFAULT,
    fit_rank_skill_map, build_initial_player_states,
    load_player_states, load_rank_map,
    STATES_PATH, MAP_PATH, DATA_DIR,
)
from MCMC.hmc_match import run_match_update, compute_feature_stats, DRAWS_DEFAULT, TUNE_DEFAULT


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n',      type=int, default=50,  help='Number of matches to process')
    p.add_argument('--draws',  type=int, default=DRAWS_DEFAULT)
    p.add_argument('--tune',   type=int, default=TUNE_DEFAULT)
    p.add_argument('--chains', type=int, default=2)
    p.add_argument('--tau_base', type=float, default=TAU_BASE_DEFAULT)
    p.add_argument('--lam',      type=float, default=LAM_DEFAULT)
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


# ── Prototype runner ──────────────────────────────────────────────────────────

def run_prototype(
    df:          pd.DataFrame,
    states:      dict[str, PlayerState],
    n_matches:   int   = 50,
    draws:       int   = DRAWS_DEFAULT,
    tune:        int   = TUNE_DEFAULT,
    chains:      int   = 2,
    tau_base:    float = TAU_BASE_DEFAULT,
    lam:         float = LAM_DEFAULT,
    verbose:     bool  = True,
) -> tuple[list, list, dict]:
    """
    Process the first n_matches rows of the preprocessed dataframe.

    Returns
    -------
    feature_rows : list[dict]
        One entry per processed match.  Includes uncertainty statistics
        (Section 1.7 of spec) plus match metadata.
    diagnostics  : list[dict]
        One entry per match.  Includes rhat_ok, ess_ok, n_div, wall_time_s.
    final_states : dict[str -> PlayerState]
        Player states after all n_matches have been processed.
    """
    feature_rows = []
    diagnostics  = []
    states_live  = copy.deepcopy(states)   # work on a copy

    # Pre-check surfaces present in the data subset
    subset = df.head(n_matches)
    print(f"\nProcessing first {n_matches} matches ...")
    print(f"  draws={draws}  tune={tune}  chains={chains}")
    print(f"  tau_base={tau_base}  lam={lam}")
    print("-" * 70)

    n_rhat_fail = 0
    n_ess_fail  = 0
    n_div_total = 0
    times       = []

    for match_idx, row in enumerate(subset.itertuples(index=False)):
        winner  = row.winner_name
        loser   = row.loser_name
        surface = str(row.surface).lower()

        # Skip if surface not in {clay, hard, grass}
        if surface not in SURF_IDX:
            if verbose:
                print(f"  [{match_idx+1:4d}] SKIP  surface='{surface}' not in model")
            continue

        # Ensure both players have a state (should always be true after init)
        if winner not in states_live or loser not in states_live:
            if verbose:
                print(f"  [{match_idx+1:4d}] SKIP  missing state for {winner} or {loser}")
            continue

        state_A = states_live[winner]
        state_B = states_live[loser]

        if verbose:
            print(f"  [{match_idx+1:4d}]  {winner[:20]:<20} vs {loser[:20]:<20}  "
                  f"surface={surface}  round={row.round}  tier={row.tourney_level}", end='  ')

        # Run HMC update
        result = run_match_update(
            state_A      = state_A,
            state_B      = state_B,
            surface      = surface,
            draws        = draws,
            tune         = tune,
            chains       = chains,
            tau_base     = tau_base,
            lam          = lam,
            random_seed  = match_idx,
        )

        # Update live states
        states_live[winner] = result.state_A
        states_live[loser]  = result.state_B

        # Compute uncertainty statistics
        stats = compute_feature_stats(result.p_samples)

        # Posterior skill means on the match surface
        surf_idx = SURF_IDX[surface]
        mu_A_prior = float(state_A.mean[surf_idx])
        mu_B_prior = float(state_B.mean[surf_idx])
        mu_A_post  = float(result.state_A.mean[surf_idx])
        mu_B_post  = float(result.state_B.mean[surf_idx])
        delta_A    = mu_A_post - mu_A_prior   # should be positive (winner updated up)
        delta_B    = mu_B_post - mu_B_prior   # should be negative (loser updated down)

        # Record feature row
        feature_row = {
            'match_idx':    match_idx,
            'tourney_id':   row.tourney_id if hasattr(row, 'tourney_id') else '',
            'tourney_name': row.tourney_name,
            'round':        row.round,
            'tourney_level':row.tourney_level,
            'surface':      surface,
            'winner_name':  winner,
            'loser_name':   loser,
            'winner_rank':  row.winner_rank,
            'loser_rank':   row.loser_rank,
            'is_upset_rank':row.is_upset_rank,
            'pi':           row.pi,
            'upset_rate':   row.upset_rate,
            # Stage 1B uncertainty stats
            **stats,
            # Skill delta (diagnostic)
            'delta_mu_winner': delta_A,
            'delta_mu_loser':  delta_B,
            'mu_winner_prior': mu_A_prior,
            'mu_loser_prior':  mu_B_prior,
            # Raw sample array stored as compact float32
            'p_samples':    result.p_samples,
        }
        feature_rows.append(feature_row)

        # Record diagnostics
        diag = {
            'match_idx':    match_idx,
            'wall_time_s':  result.wall_time_s,
            'rhat_ok':      result.rhat_ok,
            'ess_ok':       result.ess_ok,
            'n_divergences':result.n_divergences,
            'mu_pred':      stats['mu_pred'],
        }
        diagnostics.append(diag)
        times.append(result.wall_time_s)

        if not result.rhat_ok:  n_rhat_fail += 1
        if not result.ess_ok:   n_ess_fail  += 1
        n_div_total += result.n_divergences

        if verbose:
            diag_str = ' '.join([
                'rhat=' + ('OK' if result.rhat_ok else 'FAIL'),
                'ess='  + ('OK' if result.ess_ok  else 'FAIL'),
                f'div={result.n_divergences}',
                f'mu_pred={stats["mu_pred"]:.2f}',
                f'delta_W={delta_A:+.3f}',
                f't={result.wall_time_s:.1f}s',
            ])
            print(diag_str)

    n_processed = len(diagnostics)
    if n_processed == 0:
        print("No matches processed!")
        return feature_rows, diagnostics, states_live

    times_arr = np.array(times)
    print("\n" + "=" * 70)
    print("PROTOTYPE SUMMARY")
    print("=" * 70)
    print(f"  Matches processed     : {n_processed}")
    print(f"  R-hat failures        : {n_rhat_fail} / {n_processed}")
    print(f"  ESS failures          : {n_ess_fail} / {n_processed}")
    print(f"  Total divergences     : {n_div_total}")
    print(f"  Wall time: mean={times_arr.mean():.2f}s  "
          f"median={np.median(times_arr):.2f}s  "
          f"max={times_arr.max():.2f}s")
    print(f"  Estimated full pipeline ({27683} matches @ "
          f"{np.median(times_arr):.1f}s): "
          f"{27683 * np.median(times_arr) / 3600:.1f}h")

    # Posterior direction check (winner should move up, loser down)
    winner_deltas = [r['delta_mu_winner'] for r in feature_rows]
    loser_deltas  = [r['delta_mu_loser']  for r in feature_rows]
    pct_correct_w = 100 * sum(d > 0 for d in winner_deltas) / n_processed
    pct_correct_l = 100 * sum(d < 0 for d in loser_deltas)  / n_processed
    print(f"\n  Posterior direction check:")
    print(f"    Winner skill moved UP   : {pct_correct_w:.0f}% of matches")
    print(f"    Loser  skill moved DOWN : {pct_correct_l:.0f}% of matches")
    print(f"    (Expected: >50%, with full effect only on match surface)")

    return feature_rows, diagnostics, states_live


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("PHASE 4.1 — STAGE 1B PROTOTYPE")
    print("=" * 70)

    # Load or build player states
    if STATES_PATH.exists() and MAP_PATH.exists():
        print("\nLoading cached player states from prior_init ...")
        states = load_player_states()
        b, C   = load_rank_map()
        print(f"  {len(states):,} player states loaded")
    else:
        print("\nBuilding player states (prior_init not yet run) ...")
        df_tmp = pd.read_csv(DATA_DIR / 'preprocessed.csv', low_memory=False)
        df_tmp.columns = df_tmp.columns.str.strip()
        b, C   = fit_rank_skill_map(df_tmp)
        states = build_initial_player_states(df_tmp, b, C)
        from MCMC.prior_init import save_player_states, save_rank_map
        save_rank_map(b, C)
        save_player_states(states)

    print("\nLoading preprocessed.csv ...")
    df = pd.read_csv(DATA_DIR / 'preprocessed.csv', low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  {len(df):,} matches available")

    # Run prototype
    feature_rows, diagnostics, final_states = run_prototype(
        df        = df,
        states    = states,
        n_matches = args.n,
        draws     = args.draws,
        tune      = args.tune,
        chains    = args.chains,
        tau_base  = args.tau_base,
        lam       = args.lam,
        verbose   = True,
    )

    # Save outputs
    cache_path = DATA_DIR / 'prototype_cache.pkl'
    payload = {
        'feature_rows':  feature_rows,
        'diagnostics':   diagnostics,
        'player_states': final_states,
        'config': {
            'n_matches': args.n,
            'draws':     args.draws,
            'tune':      args.tune,
            'chains':    args.chains,
            'tau_base':  args.tau_base,
            'lam':       args.lam,
        }
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n  Saved prototype cache -> {cache_path}")

    # Quick sanity table
    if feature_rows:
        summary_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'p_samples'}
            for r in feature_rows[:10]
        ])
        print("\nFirst 10 matches — feature snapshot:")
        cols = ['match_idx', 'winner_name', 'loser_name', 'surface',
                'mu_pred', 'var_epist', 'var_aleat', 'delta_mu_winner']
        print(summary_df[cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("PROTOTYPE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
