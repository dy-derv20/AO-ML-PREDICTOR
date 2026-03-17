"""
Phase 2: Stage 0 Preprocessing Pipeline
Produces: tennis_data/derived_data/preprocessed.csv
          tennis_data/derived_data/upset_rate_matrix.csv
          tennis_data/derived_data/pressure_coefficients.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

INPUT  = Path('tennis_data/derived_data/complete_clean.csv')
OUTPUT = Path('tennis_data/derived_data/preprocessed.csv')
UPSET_RATE_OUTPUT = Path('tennis_data/derived_data/upset_rate_matrix.csv')
PRESSURE_OUTPUT   = Path('tennis_data/derived_data/pressure_coefficients.csv')

# Round ordering for sort and batch grouping
ROUND_ORDER = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
               'QF': 5, 'SF': 6, 'F': 7, 'RR': 3}

RECENCY_WINDOW_DAYS = 730   # 2 years for recency-filtered H2H
PRESSURE_MIN_N      = 30   # matches required for full weight on cell-specific π estimate
PRESSURE_ROUNDS     = {'SF', 'F'}
PRESSURE_TIERS      = {'1000', 'ATP Finals', 'Grand Slam'}


# ── Load ─────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading complete_clean.csv ...")
    df = pd.read_csv(INPUT, low_memory=False)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)  # preliminary sort; final sort applied after round_order is added
    print(f"  Loaded: {len(df):,} rows")
    return df


# ── 2.1 Upset rate matrix ρ_{s, tourney, r} ──────────────────────────────────

def compute_upset_rate_matrix(df):
    """
    Bootstrap approximation: upset = lower-ranked player beats higher-ranked.
    Grouping:
      - 500 / 1000 / Grand Slam / ATP Finals: per (surface, tourney_name, round)
      - 250: pooled across all 250s per (surface, round) — too few matches per
        individual tournament to produce stable rates
    """
    print("\nStep 2.1 -- Computing upset rate matrix rho_{s, tourney, r} ...")

    df['is_upset_rank'] = df['winner_rank'] > df['loser_rank']

    records = []

    # Non-250: individual tournament resolution
    for (surface, tourney_name, tourney_level, round_), group in (
        df[df['tourney_level'] != '250']
        .groupby(['surface', 'tourney_name', 'tourney_level', 'round'])
    ):
        total  = len(group)
        upsets = group['is_upset_rank'].sum()
        records.append({
            'surface':       surface,
            'tourney_name':  tourney_name,
            'tourney_level': tourney_level,
            'round':         round_,
            'total_matches': total,
            'upsets':        upsets,
            'upset_rate':    round(upsets / total, 4) if total > 0 else np.nan,
        })

    # 250s: pooled
    for (surface, round_), group in (
        df[df['tourney_level'] == '250']
        .groupby(['surface', 'round'])
    ):
        total  = len(group)
        upsets = group['is_upset_rank'].sum()
        records.append({
            'surface':       surface,
            'tourney_name':  'ATP 250 (pooled)',
            'tourney_level': '250',
            'round':         round_,
            'total_matches': total,
            'upsets':        upsets,
            'upset_rate':    round(upsets / total, 4) if total > 0 else np.nan,
        })

    matrix = pd.DataFrame(records).sort_values(
        ['tourney_level', 'surface', 'tourney_name', 'round']
    ).reset_index(drop=True)
    matrix.to_csv(UPSET_RATE_OUTPUT, index=False)

    print(f"  Computed {len(matrix)} (surface, tourney_name, round) cells")
    print(f"  Saved: {UPSET_RATE_OUTPUT}")
    print("\n  Sample upset rates:")
    sample = matrix[matrix['round'].isin(['QF', 'SF', 'F'])].sort_values('upset_rate', ascending=False).head(10)
    print(sample[['surface', 'tourney_name', 'round', 'total_matches', 'upset_rate']].to_string(index=False))

    return matrix


# ── 2.2 Career match counters n_i(t) ─────────────────────────────────────────

def compute_career_counters(df):
    """
    For each match, record how many career matches each player has played
    BEFORE this match. Then increment.
    """
    print("\nStep 2.2 -- Computing career match counters n_i(t) ...")

    career_counts = defaultdict(int)
    winner_counts = []
    loser_counts  = []

    for _, row in df.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        winner_counts.append(career_counts[w])
        loser_counts.append(career_counts[l])
        career_counts[w] += 1
        career_counts[l] += 1

    df['n_winner'] = winner_counts
    df['n_loser']  = loser_counts

    print(f"  Unique players tracked: {len(career_counts):,}")
    print(f"  Max career matches in window: {max(career_counts.values())}")
    return df


# ── 2.3 H2H records ──────────────────────────────────────────────────────────

def compute_h2h(df):
    """
    For each match, compute H2H stats strictly from prior matches.
    Four variants:
      - overall
      - surface-specific
      - pressure-context (SF/F or 1000+ tier)
      - recency-filtered (last 2 years)
    All computed as winner's win rate vs loser, from winner's perspective.
    """
    print("\nStep 2.3 -- Computing H2H records ...")

    # h2h[(A, B)] = {'wins': int, 'losses': int} from A's perspective
    h2h_overall   = defaultdict(lambda: defaultdict(int))
    h2h_surface   = defaultdict(lambda: defaultdict(int))
    h2h_pressure  = defaultdict(lambda: defaultdict(int))
    # Recency needs history of (date, winner, loser) to filter by window
    h2h_history   = []  # list of (date, winner, loser, surface, is_pressure)

    cols = {
        'h2h_overall_w':      [],
        'h2h_overall_l':      [],
        'h2h_surface_w':      [],
        'h2h_surface_l':      [],
        'h2h_pressure_w':     [],
        'h2h_pressure_l':     [],
        'h2h_recency_w':      [],
        'h2h_recency_l':      [],
    }

    for _, row in df.iterrows():
        w       = row['winner_name']
        l       = row['loser_name']
        surf    = row['surface']
        date    = row['tourney_date']
        tier    = row['tourney_level']
        rnd     = row['round']
        is_pres = (rnd in PRESSURE_ROUNDS) or (tier in PRESSURE_TIERS)

        # -- Overall H2H (from winner's perspective vs loser)
        cols['h2h_overall_w'].append(h2h_overall[(w, l)]['wins'])
        cols['h2h_overall_l'].append(h2h_overall[(w, l)]['losses'])

        # -- Surface H2H
        cols['h2h_surface_w'].append(h2h_surface[(w, l, surf)]['wins'])
        cols['h2h_surface_l'].append(h2h_surface[(w, l, surf)]['losses'])

        # -- Pressure H2H
        cols['h2h_pressure_w'].append(h2h_pressure[(w, l)]['wins'])
        cols['h2h_pressure_l'].append(h2h_pressure[(w, l)]['losses'])

        # -- Recency H2H: count wins/losses in last 2 years from history
        cutoff = date - timedelta(days=RECENCY_WINDOW_DAYS)
        rec_wins   = sum(1 for d, hw, hl, _, _ in h2h_history
                         if d >= cutoff and hw == w and hl == l)
        rec_losses = sum(1 for d, hw, hl, _, _ in h2h_history
                         if d >= cutoff and hw == l and hl == w)
        cols['h2h_recency_w'].append(rec_wins)
        cols['h2h_recency_l'].append(rec_losses)

        # -- Update all records AFTER recording pre-match state
        h2h_overall[(w, l)]['wins']   += 1
        h2h_overall[(l, w)]['losses'] += 1

        h2h_surface[(w, l, surf)]['wins']   += 1
        h2h_surface[(l, w, surf)]['losses'] += 1

        if is_pres:
            h2h_pressure[(w, l)]['wins']   += 1
            h2h_pressure[(l, w)]['losses'] += 1

        h2h_history.append((date, w, l, surf, is_pres))

    for col, values in cols.items():
        df[col] = values

    print(f"  H2H columns added: {list(cols.keys())}")
    return df


# ── 2.4 Pressure coefficients π(r, tier) ─────────────────────────────────────

def compute_pressure_coefficients(df):
    """
    Empirical pressure coefficients π(r, tier) estimated from upset rate data.

    Signal: how much does the upset rate at (r, tier) deviate from the global
    baseline? Higher relative upset rate = more unpredictable context = higher π.

    Hierarchical structure:
      - Cell-specific: π_cell(r, tier) = upset_rate(r, tier) / p0
      - Fallback:      π_fallback(r, tier) = geometric_mean(π_r(r), π_t(tier))
                       where π_r and π_t are round-only and tier-only marginals.
                       This is still parameterized by both r and tier but estimated
                       from more data, so it is more stable for sparse cells.
      - Shrinkage:     w = min(n(r,tier) / PRESSURE_MIN_N, 1.0)
                       π_raw = w * π_cell + (1-w) * π_fallback
      - Normalization: scale so minimum cell = 1.0
    """
    print("\nStep 2.4 -- Computing empirical pressure coefficients pi(r, tier) ...")

    p0 = df['is_upset_rank'].mean()  # global baseline upset rate

    # Round-only and tier-only marginal upset rates
    round_rates = df.groupby('round')['is_upset_rank'].mean()
    tier_rates  = df.groupby('tourney_level')['is_upset_rank'].mean()

    # Cell-level aggregation
    cell = (
        df.groupby(['round', 'tourney_level'])['is_upset_rank']
        .agg(n_matches='count', cell_upsets='sum')
        .reset_index()
    )
    cell['cell_rate'] = cell['cell_upsets'] / cell['n_matches']

    records = []
    for _, row in cell.iterrows():
        r    = row['round']
        tier = row['tourney_level']
        n    = row['n_matches']

        pi_cell = row['cell_rate'] / p0

        # Factored fallback: geometric mean of marginal coefficients
        pi_r        = round_rates.get(r, p0) / p0
        pi_t        = tier_rates.get(tier, p0) / p0
        pi_fallback = np.sqrt(pi_r * pi_t)

        w      = min(n / PRESSURE_MIN_N, 1.0)
        pi_raw = w * pi_cell + (1 - w) * pi_fallback

        records.append({
            'round':            r,
            'tourney_level':    tier,
            'n_matches':        int(n),
            'cell_upset_rate':  round(row['cell_rate'], 4),
            'pi_cell':          round(pi_cell, 4),
            'pi_fallback':      round(pi_fallback, 4),
            'shrinkage_weight': round(w, 4),
            'pi_raw':           round(pi_raw, 4),
        })

    coeff_df = pd.DataFrame(records)
    min_pi = coeff_df['pi_raw'].min()
    coeff_df['pi'] = (coeff_df['pi_raw'] / min_pi).round(4)

    coeff_df = coeff_df.sort_values(['tourney_level', 'round']).reset_index(drop=True)
    coeff_df.to_csv(PRESSURE_OUTPUT, index=False)

    print(f"  Global baseline upset rate p0: {p0:.4f}")
    print(f"  pi range: {coeff_df['pi'].min():.2f} -- {coeff_df['pi'].max():.2f}")
    print(f"  Saved: {PRESSURE_OUTPUT}")
    print("\n  Sample coefficients (QF / SF / F):")
    sample = coeff_df[coeff_df['round'].isin(['QF', 'SF', 'F'])].sort_values('pi', ascending=False).head(12)
    print(sample[['tourney_level', 'round', 'n_matches', 'cell_upset_rate', 'shrinkage_weight', 'pi']].to_string(index=False))

    return coeff_df


# ── 2.5 Merge upset rate and save ────────────────────────────────────────────

def merge_and_save(df, coeff_df):
    print("\nStep 2.5 -- Merging upset rates, pressure coefficients, and saving ...")

    upset_matrix = pd.read_csv(UPSET_RATE_OUTPUT)

    # 250 rows look up the pooled entry; all others look up by tourney_name
    df['_lookup_name'] = df['tourney_name']
    df.loc[df['tourney_level'] == '250', '_lookup_name'] = 'ATP 250 (pooled)'

    lookup = upset_matrix[['surface', 'tourney_name', 'round', 'upset_rate']].rename(
        columns={'tourney_name': '_lookup_name'}
    )
    df = df.merge(lookup, on=['surface', '_lookup_name', 'round'], how='left')
    df.drop(columns=['_lookup_name'], inplace=True)

    missing_rho = df['upset_rate'].isna().sum()
    if missing_rho > 0:
        print(f"  WARNING: {missing_rho} rows missing upset_rate -- assigning global mean")
        df['upset_rate'] = df['upset_rate'].fillna(df['upset_rate'].mean())

    # Merge pressure coefficients
    df = df.merge(
        coeff_df[['round', 'tourney_level', 'pi']],
        on=['round', 'tourney_level'],
        how='left'
    )
    missing_pi = df['pi'].isna().sum()
    if missing_pi > 0:
        print(f"  WARNING: {missing_pi} rows missing pi -- assigning 1.0")
        df['pi'] = df['pi'].fillna(1.0)

    # Add batch grouping columns
    df['round_order'] = df['round'].map(ROUND_ORDER)
    df['week_id'] = pd.to_datetime(df['tourney_date']).dt.strftime('%G-W%V')

    # Final sort: cross-week → within-week by tournament → within-tournament by round → within-round by match
    df = df.sort_values(['tourney_date', 'tourney_id', 'round_order', 'match_num']).reset_index(drop=True)

    df.to_csv(OUTPUT, index=False)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)

    print(f"  Saved: {OUTPUT}  ({size_mb:.1f} MB)")
    print(f"\n=== FINAL PREPROCESSED DATASET ===")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Columns added this phase: is_upset_rank, n_winner, n_loser,")
    print(f"    h2h_overall_w/l, h2h_surface_w/l, h2h_pressure_w/l,")
    print(f"    h2h_recency_w/l, upset_rate, pi, round_order, week_id")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 2: STAGE 0 PREPROCESSING")
    print("=" * 60)

    df           = load_data()
    _            = compute_upset_rate_matrix(df)
    coeff_df     = compute_pressure_coefficients(df)
    df           = compute_career_counters(df)
    df           = compute_h2h(df)
    merge_and_save(df, coeff_df)

    print("\n" + "=" * 60)
    print("DONE -- preprocessed.csv ready for Stage 1A")
    print("=" * 60)


if __name__ == '__main__':
    main()
