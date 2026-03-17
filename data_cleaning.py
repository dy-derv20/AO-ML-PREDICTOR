"""
Phase 1: Data Cleaning
Produces: tennis_data/derived_data/complete_clean.csv
"""

import pandas as pd
from pathlib import Path

INPUT  = Path('tennis_data/derived_data/complete.csv')
OUTPUT = Path('tennis_data/derived_data/complete_clean.csv')

RANKINGS_FILES = {
    '10s': Path('tennis_data/match_by_match/atp_rankings_10s.csv'),
    '20s': Path('tennis_data/match_by_match/atp_rankings_20s.csv'),
    'cur': Path('tennis_data/match_by_match/atp_rankings_current.csv'),
}
PLAYERS_FILE = Path('tennis_data/match_by_match/atp_players.csv')

PLACEHOLDER_RANK = 1500

# ATP 500 tournament names (union of TML labels + historical knowledge)
ATP_500_NAMES = {
    'Rotterdam', 'Dubai', 'Acapulco', 'Barcelona', 'Hamburg',
    'Halle', "Queen's Club", 'Washington', 'Beijing', 'Tokyo',
    'Basel', 'Vienna', 'Rio de Janeiro', 'ATP Rio de Janeiro',
    'Rio De Janeiro', 'Dallas', 'Munich', 'Doha',
}

# Events that are not individual ATP tour matches — drop entirely
DROP_TOURNAMENT_NAMES = {
    'Laver Cup', 'Atp Cup', 'United Cup',
    'Next Gen ATP Finals', 'NextGen Finals', 'Next Gen Finals',
    'Rio Olympics', 'Tokyo Olympics',
}

LEVEL_MAP = {
    'G': 'Grand Slam',
    'M': '1000',
    'F': 'ATP Finals',
}


# ── Step 1: Load ─────────────────────────────────────────────────────────────

def load_data():
    print("Loading complete.csv ...")
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"  Loaded: {len(df):,} rows")
    return df


# ── Step 2: Drop non-ATP events ───────────────────────────────────────────────

def drop_non_atp(df):
    print("\nStep 1 — Dropping non-ATP events ...")
    before = len(df)

    # Davis Cup and Olympics by level code
    df = df[~df['tourney_level'].isin(['D', 'O'])]

    # Team/exhibition events embedded in 'A' level by name
    df = df[~df['tourney_name'].isin(DROP_TOURNAMENT_NAMES)]

    dropped = before - len(df)
    print(f"  Dropped: {dropped:,} rows  |  {len(df):,} remaining")
    return df.reset_index(drop=True)


# ── Step 3: Standardize tourney_level ────────────────────────────────────────

def standardize_levels(df):
    print("\nStep 2 — Standardizing tourney_level ...")

    def map_level(row):
        lvl  = row['tourney_level']
        name = row['tourney_name']
        if lvl in LEVEL_MAP:
            return LEVEL_MAP[lvl]
        if lvl in ('250', '500'):      # TML already correct
            return lvl
        if lvl == 'A':
            return '500' if name in ATP_500_NAMES else '250'
        return lvl                     # fallback — will surface in validation

    df['tourney_level'] = df.apply(map_level, axis=1)
    print("  Distribution after mapping:")
    for lvl, cnt in df['tourney_level'].value_counts().items():
        print(f"    {lvl}: {cnt:,}")
    return df


# ── Step 4: Fix null surfaces ─────────────────────────────────────────────────

def fix_surfaces(df):
    print("\nStep 3 — Checking null surfaces ...")
    null_surf = df['surface'].isna().sum()
    if null_surf == 0:
        print("  No null surfaces — nothing to fix.")
        return df

    print(f"  {null_surf} null surfaces found:")
    print(df[df['surface'].isna()][['tourney_name', 'tourney_date']].to_string())

    # Known surface by tournament name for any stragglers
    surface_lookup = {
        'Wimbledon': 'Grass', 'Roland Garros': 'Clay',
        'Australian Open': 'Hard', 'US Open': 'Hard',
    }
    def impute_surface(row):
        if pd.isna(row['surface']):
            for key, surf in surface_lookup.items():
                if key in str(row['tourney_name']):
                    return surf
            return 'Hard'   # safe default — flag for manual review
        return row['surface']

    df['surface'] = df.apply(impute_surface, axis=1)
    print(f"  Remaining nulls after imputation: {df['surface'].isna().sum()}")
    return df


# ── Step 5: Fix null loser_id ─────────────────────────────────────────────────

def fix_loser_id(df):
    print("\nStep 4 — Fixing null loser_id ...")
    null_mask = df['loser_id'].isna()
    if null_mask.sum() == 0:
        print("  No null loser_ids — nothing to fix.")
        return df

    players = pd.read_csv(PLAYERS_FILE, low_memory=False)
    player_id_map = {
        f"{r['name_first']} {r['name_last']}": r['player_id']
        for _, r in players.iterrows()
    }

    fixed = 0
    for idx in df[null_mask].index:
        name = df.at[idx, 'loser_name']
        if name in player_id_map:
            df.at[idx, 'loser_id'] = str(player_id_map[name])
            print(f"  Fixed: '{name}' -> {player_id_map[name]}")
            fixed += 1
        else:
            print(f"  WARNING: Could not resolve loser_id for '{name}'")
    print(f"  Fixed {fixed} / {null_mask.sum()} null loser_ids")
    return df


# ── Step 6: Resolve null ranks ────────────────────────────────────────────────

def load_rankings():
    """Load and combine all rankings files into one sorted DataFrame."""
    print("\n  Loading ATP rankings files ...")
    frames = []
    for label, path in RANKINGS_FILES.items():
        r = pd.read_csv(path)
        r.columns = ['ranking_date', 'rank', 'player_id', 'ranking_points']
        frames.append(r)
    rankings = pd.concat(frames, ignore_index=True)
    rankings['ranking_date'] = pd.to_datetime(rankings['ranking_date'].astype(str), format='%Y%m%d')
    rankings = rankings.sort_values('ranking_date')
    return rankings


def lookup_rank(player_id, match_date, rankings):
    """Return the most recent rank for player_id at or before match_date."""
    subset = rankings[(rankings['player_id'] == player_id) &
                      (rankings['ranking_date'] <= match_date)]
    if subset.empty:
        return None
    return int(subset.iloc[-1]['rank'])


def fix_ranks(df):
    print("\nStep 5 — Resolving null ranks ...")
    winner_nulls = df['winner_rank'].isna().sum()
    loser_nulls  = df['loser_rank'].isna().sum()
    print(f"  Null winner_rank: {winner_nulls}  |  Null loser_rank: {loser_nulls}")

    df['rank_imputed'] = False
    rankings = load_rankings()

    df['tourney_date_dt'] = pd.to_datetime(df['tourney_date'].astype(str), format='%Y%m%d')

    def resolve(row, player_col, rank_col):
        if pd.notna(row[rank_col]):
            return row[rank_col], False
        pid = row[player_col]
        if pd.notna(pid):
            try:
                found = lookup_rank(int(pid), row['tourney_date_dt'], rankings)
                if found:
                    return found, True
            except (ValueError, TypeError):
                pass  # non-numeric TML player ID — fall through to placeholder
        return PLACEHOLDER_RANK, True

    winner_resolved = df.apply(lambda r: resolve(r, 'winner_id', 'winner_rank'), axis=1)
    loser_resolved  = df.apply(lambda r: resolve(r, 'loser_id',  'loser_rank'),  axis=1)

    df['winner_rank']  = [x[0] for x in winner_resolved]
    df['loser_rank']   = [x[0] for x in loser_resolved]
    df['rank_imputed'] = (
        pd.Series([x[1] for x in winner_resolved], index=df.index) |
        pd.Series([x[1] for x in loser_resolved],  index=df.index)
    )

    df = df.drop(columns=['tourney_date_dt'])

    remaining_w = df['winner_rank'].isna().sum()
    remaining_l = df['loser_rank'].isna().sum()
    imputed     = df['rank_imputed'].sum()
    print(f"  Remaining null winner_rank: {remaining_w}")
    print(f"  Remaining null loser_rank:  {remaining_l}")
    print(f"  Rows with at least one imputed rank: {imputed}")
    return df


# ── Step 7: Chronological sort and save ──────────────────────────────────────

def sort_and_save(df):
    print("\nStep 6 — Sorting chronologically and saving ...")
    df['tourney_date'] = df['tourney_date'].astype(str)
    df = df.sort_values(['tourney_date', 'match_num'], ascending=True).reset_index(drop=True)

    df.to_csv(OUTPUT, index=False)
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"  Saved: {OUTPUT}  ({size_mb:.1f} MB)")

    print("\n=== FINAL DATASET STATS ===")
    print(f"  Total rows:        {len(df):,}")
    print(f"  Columns:           {len(df.columns)}")
    print(f"  Null surfaces:     {df['surface'].isna().sum()}")
    print(f"  Null winner_rank:  {df['winner_rank'].isna().sum()}")
    print(f"  Null loser_rank:   {df['loser_rank'].isna().sum()}")
    print(f"  Null loser_id:     {df['loser_id'].isna().sum()}")
    print(f"  Imputed ranks:     {df['rank_imputed'].sum()}")
    print(f"\n  tourney_level distribution:")
    for lvl, cnt in df['tourney_level'].value_counts().items():
        print(f"    {lvl}: {cnt:,}")
    print(f"\n  Year distribution:")
    years = df['tourney_date'].astype(str).str[:4].value_counts().sort_index()
    for yr, cnt in years.items():
        print(f"    {yr}: {cnt:,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1: DATA CLEANING")
    print("=" * 60)

    df = load_data()
    df = drop_non_atp(df)
    df = standardize_levels(df)
    df = fix_surfaces(df)
    df = fix_loser_id(df)
    df = fix_ranks(df)
    sort_and_save(df)

    print("\n" + "=" * 60)
    print("DONE — complete_clean.csv ready for Phase 2")
    print("=" * 60)


if __name__ == '__main__':
    main()
