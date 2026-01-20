"""
Precise Tennis Data Merger
Merges TML 2025/2026 data with Jeff Sackmann 2015-2024 data

Column Differences:
- TML has 'indoor' column (Sackmann doesn't)
- TML has winner_rank and loser_rank BEFORE match stats (Sackmann has them AFTER)
- All other columns are identical in name and order

Usage:
    python merge_precise.py
"""

import pandas as pd
from pathlib import Path

# Configuration
DATA_DIR = Path('tennis_data')
SACKMANN_YEARS = range(2015, 2025)  # 2015-2024

# Define Sackmann column order (49 columns)
SACKMANN_COLUMNS = [
    'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 
    'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry', 
    'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 
    'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand', 
    'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round', 
    'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
    'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 
    'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'winner_rank', 
    'winner_rank_points', 'loser_rank', 'loser_rank_points'
]

def load_sackmann_data():
    """Load all Sackmann match files (2015-2024)"""
    print("="*70)
    print("LOADING SACKMANN DATA (2015-2024)")
    print("="*70 + "\n")
    
    all_dataframes = []
    total_matches = 0
    
    for year in SACKMANN_YEARS:
        filepath = DATA_DIR / f'atp_matches_{year}.csv'
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            matches = len(df)
            total_matches += matches
            print(f"✓ {year}: {matches:,} matches")
            all_dataframes.append(df)
        else:
            print(f"✗ {year}: File not found - {filepath}")
    
    if not all_dataframes:
        print("\n✗ ERROR: No Sackmann files loaded!")
        return None
    
    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✓ Total Sackmann matches (2015-2024): {total_matches:,}")
    
    return combined

def convert_tml_to_sackmann_format(tml_df, year):
    """Convert TML format to Sackmann format"""
    print(f"\nConverting TML {year} to Sackmann format...")
    print(f"  Original columns: {len(tml_df.columns)}")
    
    # TML has 50 columns (includes 'indoor')
    # Sackmann has 49 columns (no 'indoor')
    
    # Drop 'indoor' column if it exists
    if 'indoor' in tml_df.columns:
        tml_df = tml_df.drop(columns=['indoor'])
        print(f"  ✓ Dropped 'indoor' column")
    
    # TML column order (after dropping 'indoor'):
    # tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date, 
    # match_num, winner_id, winner_seed, winner_entry, winner_name, winner_hand, 
    # winner_ht, winner_ioc, winner_age, winner_rank, winner_rank_points, loser_id, 
    # loser_seed, loser_entry, loser_name, loser_hand, loser_ht, loser_ioc, 
    # loser_age, loser_rank, loser_rank_points, score, best_of, round, minutes, 
    # w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, 
    # w_bpFaced, l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, 
    # l_bpSaved, l_bpFaced
    
    # We need to reorder to match Sackmann:
    # Move winner_rank, winner_rank_points, loser_rank, loser_rank_points to the end
    
    # Get current columns
    current_cols = tml_df.columns.tolist()
    
    # Define the reordering
    new_order = [
        # First 15 columns stay same
        'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
        'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
        'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
        # Skip winner_rank and winner_rank_points for now
        'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
        'loser_ht', 'loser_ioc', 'loser_age',
        # Skip loser_rank and loser_rank_points for now
        'score', 'best_of', 'round', 'minutes',
        # Match stats
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_SvGms', 'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
        'l_SvGms', 'l_bpSaved', 'l_bpFaced',
        # Rankings at the end (Sackmann position)
        'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points'
    ]
    
    # Reorder columns
    tml_df = tml_df[new_order]
    print(f"  ✓ Reordered columns to match Sackmann format")
    print(f"  Final columns: {len(tml_df.columns)}")
    
    # Verify column order matches
    if tml_df.columns.tolist() == SACKMANN_COLUMNS:
        print(f"  ✓ Column order MATCHES Sackmann perfectly!")
    else:
        print(f"  ⚠ Column order mismatch detected")
        # Show differences
        for i, (tml_col, sack_col) in enumerate(zip(tml_df.columns, SACKMANN_COLUMNS)):
            if tml_col != sack_col:
                print(f"    Position {i}: TML='{tml_col}' vs Sackmann='{sack_col}'")
    
    return tml_df

def load_and_convert_tml_file(filename, year):
    """Load TML file and convert to Sackmann format"""
    print("\n" + "="*70)
    print(f"LOADING TML {year} DATA")
    print("="*70 + "\n")
    
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        print(f"✗ File not found: {filepath}")
        return None
    
    # Load file
    df = pd.read_csv(filepath)
    print(f"✓ Loaded: {len(df):,} matches")
    print(f"  Columns: {len(df.columns)}")
    
    # Convert format
    df = convert_tml_to_sackmann_format(df, year)
    
    # Validate data
    validate_tml_data(df, year)
    
    return df

def validate_tml_data(df, year):
    """Validate TML data"""
    print(f"\nValidating {year} data:")
    
    # Check date range
    if 'tourney_date' in df.columns:
        min_date = df['tourney_date'].min()
        max_date = df['tourney_date'].max()
        print(f"  Date range: {min_date} to {max_date}")
    
    # Check for Australian Open (if 2025)
    if year == 2025 and 'tourney_name' in df.columns:
        ao = df[df['tourney_name'].str.contains('Australian', na=False, case=False)]
        if len(ao) > 0:
            print(f"  ✓ Australian Open 2025: {len(ao)} matches found ⭐")
            
            # Check final
            final = ao[ao['round'] == 'F']
            if len(final) > 0:
                winner = final.iloc[0]['winner_name']
                loser = final.iloc[0]['loser_name']
                score = final.iloc[0]['score']
                print(f"    Champion: {winner}")
                print(f"    Runner-up: {loser}")
                print(f"    Score: {score}")
        else:
            print(f"  ⚠ Australian Open 2025: NOT FOUND")
    
    # Check for key columns
    null_counts = df[['winner_name', 'loser_name', 'score']].isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ⚠ Null values detected:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"    {col}: {count}")
    else:
        print(f"  ✓ No null values in key columns")

def merge_all_data(sackmann_df, tml_2025_df, tml_2026_df):
    """Merge all datasets"""
    print("\n" + "="*70)
    print("MERGING ALL DATA")
    print("="*70 + "\n")
    
    all_dfs = [sackmann_df]
    
    if tml_2025_df is not None:
        print(f"Adding 2025 data: {len(tml_2025_df):,} matches")
        all_dfs.append(tml_2025_df)
    
    if tml_2026_df is not None:
        print(f"Adding 2026 data: {len(tml_2026_df):,} matches")
        all_dfs.append(tml_2026_df)
    
    # Combine
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal before deduplication: {len(combined):,} matches")
    
    # Remove duplicates
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=['tourney_date', 'winner_name', 'loser_name'],
        keep='first'
    )
    after = len(combined)
    
    if before != after:
        print(f"Removed {before - after:,} duplicate matches")
    else:
        print(f"No duplicates found")
    
    print(f"Total after deduplication: {len(combined):,} matches")
    
    return combined

def validate_final_data(df):
    """Validate final merged dataset"""
    print("\n" + "="*70)
    print("FINAL DATASET VALIDATION")
    print("="*70 + "\n")
    
    # Date range
    df_temp = df.copy()
    df_temp['tourney_date'] = pd.to_datetime(
        df_temp['tourney_date'].astype(str),
        format='%Y%m%d',
        errors='coerce'
    )
    
    min_date = df_temp['tourney_date'].min()
    max_date = df_temp['tourney_date'].max()
    print(f"✓ Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Year distribution
    year_counts = df_temp['tourney_date'].dt.year.value_counts().sort_index()
    print(f"\n✓ Matches per year:")
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} matches")
    
    # Check 2025 specifically
    matches_2025 = len(df_temp[df_temp['tourney_date'].dt.year == 2025])
    matches_2026 = len(df_temp[df_temp['tourney_date'].dt.year == 2026])
    
    print(f"\n📊 NEW DATA:")
    print(f"  2025: {matches_2025:,} matches")
    print(f"  2026: {matches_2026:,} matches")
    
    if matches_2025 == 0:
        print(f"  ⚠ WARNING: No 2025 data found in final dataset!")
    if matches_2026 == 0:
        print(f"  ⚠ WARNING: No 2026 data found in final dataset!")
    
    # Australian Open 2025
    ao_2025 = df_temp[
        (df_temp['tourney_name'].str.contains('Australian', na=False, case=False)) &
        (df_temp['tourney_date'].dt.year == 2025)
    ]
    
    if len(ao_2025) > 0:
        print(f"\n✓ Australian Open 2025: {len(ao_2025)} matches ⭐")
        final = ao_2025[ao_2025['round'] == 'F']
        if len(final) > 0:
            print(f"  Champion: {final.iloc[0]['winner_name']}")
    else:
        print(f"\n✗ Australian Open 2025: NOT FOUND IN FINAL DATASET!")
    
    # Column check
    print(f"\n✓ Total columns: {len(df.columns)}")
    print(f"  Expected: {len(SACKMANN_COLUMNS)}")
    if len(df.columns) == len(SACKMANN_COLUMNS):
        print(f"  ✓ Column count matches!")
    else:
        print(f"  ✗ Column count mismatch!")
    
    # Total stats
    print(f"\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total matches: {len(df):,}")
    unique_players = set(df['winner_name'].dropna()) | set(df['loser_name'].dropna())
    print(f"Unique players: {len(unique_players):,}")
    print(f"Unique tournaments: {df['tourney_name'].nunique():,}")
    
    return df

def save_data(df, filename='atp_matches_complete_2015-2026.csv'):
    """Save merged dataset"""
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70 + "\n")
    
    output_path = DATA_DIR / filename
    
    # Sort by date
    df = df.sort_values('tourney_date')
    
    # Save
    df.to_csv(output_path, index=False)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {file_size:.2f} MB")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    return output_path

def main():
    print("\n" + "="*70)
    print("PRECISE TENNIS DATA MERGER")
    print("Sackmann (2015-2024) + TML (2025-2026)")
    print("="*70 + "\n")
    
    # Step 1: Load Sackmann data
    sackmann_df = load_sackmann_data()
    if sackmann_df is None:
        print("\n✗ Failed to load Sackmann data. Exiting.")
        return
    
    # Step 2: Load and convert TML 2025
    tml_2025_df = load_and_convert_tml_file('2025.csv', 2025)
    
    # Step 3: Load and convert TML 2026
    tml_2026_df = load_and_convert_tml_file('2026.csv', 2026)
    
    # Check if we got any TML data
    if tml_2025_df is None and tml_2026_df is None:
        print("\n✗ No TML data loaded. Nothing to merge.")
        return
    
    # Step 4: Merge all data
    merged_df = merge_all_data(sackmann_df, tml_2025_df, tml_2026_df)
    
    # Step 5: Validate
    merged_df = validate_final_data(merged_df)
    
    # Step 6: Save
    output_path = save_data(merged_df)
    
    # Final summary
    print("\n" + "="*70)
    print("✓ MERGE COMPLETE!")
    print("="*70)
    print(f"\nOutput file: {output_path.name}")
    print(f"\nNext steps:")
    print(f"  1. Verify: head {output_path}")
    print(f"  2. Check 2025 data: grep 'Australian Open' {output_path} | wc -l")
    print(f"  3. Calculate ELO ratings")
    print(f"  4. Build prediction model")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()