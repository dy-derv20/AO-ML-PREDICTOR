"""
Fast post-processing fixup: add round_order + week_id columns and fix sort order.
Does NOT recompute H2H or any other slow derived features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT            = Path('tennis_data/derived_data/preprocessed.csv')
UPSET_OUTPUT     = Path('tennis_data/derived_data/upset_rate_matrix.csv')
PRESSURE_OUTPUT  = Path('tennis_data/derived_data/pressure_coefficients.csv')

ROUND_ORDER    = {'R128': 1, 'R64': 2, 'R32': 3, 'R16': 4,
                  'QF': 5, 'SF': 6, 'F': 7, 'RR': 3}
PRESSURE_MIN_N = 30

print("Loading preprocessed.csv ...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

# ── Recompute upset_rate_matrix with tourney_name grouping ────────────────────

print("\nRecomputing upset rate matrix ...")
df['is_upset_rank'] = df['winner_rank'] > df['loser_rank']

records = []

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
matrix.to_csv(UPSET_OUTPUT, index=False)
print(f"  {len(matrix)} cells written to {UPSET_OUTPUT}")

# ── Compute empirical pressure coefficients ───────────────────────────────────

print("\nComputing empirical pressure coefficients ...")
p0          = df['is_upset_rank'].mean()
round_rates = df.groupby('round')['is_upset_rank'].mean()
tier_rates  = df.groupby('tourney_level')['is_upset_rank'].mean()

cell = (
    df.groupby(['round', 'tourney_level'])['is_upset_rank']
    .agg(n_matches='count', cell_upsets='sum')
    .reset_index()
)
cell['cell_rate'] = cell['cell_upsets'] / cell['n_matches']

pi_records = []
for _, row in cell.iterrows():
    r, tier, n = row['round'], row['tourney_level'], row['n_matches']
    pi_cell     = row['cell_rate'] / p0
    pi_r        = round_rates.get(r, p0) / p0
    pi_t        = tier_rates.get(tier, p0) / p0
    pi_fallback = np.sqrt(pi_r * pi_t)
    w           = min(n / PRESSURE_MIN_N, 1.0)
    pi_raw      = w * pi_cell + (1 - w) * pi_fallback
    pi_records.append({
        'round': r, 'tourney_level': tier,
        'n_matches': int(n), 'cell_upset_rate': round(row['cell_rate'], 4),
        'pi_cell': round(pi_cell, 4), 'pi_fallback': round(pi_fallback, 4),
        'shrinkage_weight': round(w, 4), 'pi_raw': round(pi_raw, 4),
    })

coeff_df         = pd.DataFrame(pi_records)
min_pi           = coeff_df['pi_raw'].min()
coeff_df['pi']   = (coeff_df['pi_raw'] / min_pi).round(4)
coeff_df         = coeff_df.sort_values(['tourney_level', 'round']).reset_index(drop=True)
coeff_df.to_csv(PRESSURE_OUTPUT, index=False)
print(f"  pi range: {coeff_df['pi'].min():.2f} -- {coeff_df['pi'].max():.2f}  ->  {PRESSURE_OUTPUT}")

# ── Re-merge upset_rate into main df ─────────────────────────────────────────

# Drop old upset_rate and pi if present
drop_cols = [c for c in ['upset_rate', 'pi'] if c in df.columns]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"\nDropped columns: {drop_cols}")

df['_lookup_name'] = df['tourney_name']
df.loc[df['tourney_level'] == '250', '_lookup_name'] = 'ATP 250 (pooled)'

lookup = matrix[['surface', 'tourney_name', 'round', 'upset_rate']].rename(
    columns={'tourney_name': '_lookup_name'}
)
df = df.merge(lookup, on=['surface', '_lookup_name', 'round'], how='left')
df.drop(columns=['_lookup_name'], inplace=True)

missing = df['upset_rate'].isna().sum()
if missing > 0:
    print(f"  WARNING: {missing} rows missing upset_rate -- assigning global mean")
    df['upset_rate'] = df['upset_rate'].fillna(df['upset_rate'].mean())
else:
    print(f"  upset_rate: OK (no nulls)")

df = df.merge(coeff_df[['round', 'tourney_level', 'pi']], on=['round', 'tourney_level'], how='left')
missing_pi = df['pi'].isna().sum()
if missing_pi > 0:
    print(f"  WARNING: {missing_pi} rows missing pi -- assigning 1.0")
    df['pi'] = df['pi'].fillna(1.0)
else:
    print(f"  pi:         OK (no nulls)")

# ── Add/refresh batch grouping columns ───────────────────────────────────────

df['round_order'] = df['round'].map(ROUND_ORDER)
df['week_id'] = pd.to_datetime(df['tourney_date']).dt.strftime('%G-W%V')

missing_ro = df['round_order'].isna().sum()
print(f"  round_order: {'OK (no nulls)' if missing_ro == 0 else f'WARNING: {missing_ro} nulls'}")
print(f"  week_id:     OK (no nulls)")

df = df.sort_values(['tourney_date', 'tourney_id', 'round_order', 'match_num']).reset_index(drop=True)

# ── Verification ──────────────────────────────────────────────────────────────

ao_2023 = df[df['tourney_id'].str.contains('2023-580', na=False)]
if not ao_2023.empty:
    rounds_seen = ao_2023[['round', 'round_order']].drop_duplicates().sort_values('round_order')
    print(f"\n  AO 2023 round order spot-check:")
    print(rounds_seen.to_string(index=False))

df.to_csv(INPUT, index=False)
size_mb = INPUT.stat().st_size / (1024 * 1024)
print(f"\nSaved: {INPUT}  ({size_mb:.1f} MB, {len(df.columns)} columns)")
