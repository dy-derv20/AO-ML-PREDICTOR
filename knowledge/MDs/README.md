# ATP Tennis Match Prediction — Dataset

## Overview

Historical ATP tour match data for ML-based match outcome prediction research. The primary artifact is a single merged dataset covering 2015–2026.

## Data Structure

```
tennis_data/
├── derived_data/
│   └── complete.csv              # Primary dataset — merged 2015–2026
└── match_by_match/
    ├── atp_matches_2015.csv      # Jeff Sackmann source files (2015–2024)
    ├── ...
    ├── atp_matches_2024.csv
    ├── 2025.csv                  # TML Database
    ├── 2026.csv                  # TML Database
    ├── atp_players.csv           # Player biographical data
    ├── atp_rankings_10s.csv
    ├── atp_rankings_20s.csv
    └── atp_rankings_current.csv
```

## Primary Dataset: `complete.csv`

**Coverage:** January 2015 – early 2026
**Rows:** ~30,745 matches
**Columns:** 49
**Scope:** ATP tour-level main draw only

Column schema follows the Jeff Sackmann standard:

| Column | Description |
|--------|-------------|
| `tourney_id` | Unique tournament identifier |
| `tourney_name` | Tournament name |
| `surface` | Court surface (Hard, Clay, Grass, Carpet) |
| `draw_size` | Draw size |
| `tourney_level` | G=Grand Slam, M=Masters, A=500/250, D=Davis Cup |
| `tourney_date` | Tournament start date (YYYYMMDD) |
| `match_num` | Match number within tournament |
| `winner_id` / `loser_id` | Player ID (links to `atp_players.csv`) |
| `winner_seed` / `loser_seed` | Tournament seed |
| `winner_entry` / `loser_entry` | Entry type (Q, WC, LL, etc.) |
| `winner_name` / `loser_name` | Player name |
| `winner_hand` / `loser_hand` | Dominant hand (R/L) |
| `winner_ht` / `loser_ht` | Height (cm) |
| `winner_ioc` / `loser_ioc` | Country code |
| `winner_age` / `loser_age` | Age at tournament start |
| `score` | Match score string |
| `best_of` | Best of 3 or 5 sets |
| `round` | R128, R64, R32, R16, QF, SF, F |
| `minutes` | Match duration |
| `w_ace` / `l_ace` | Aces |
| `w_df` / `l_df` | Double faults |
| `w_svpt` / `l_svpt` | Total serve points |
| `w_1stIn` / `l_1stIn` | First serves in |
| `w_1stWon` / `l_1stWon` | First serve points won |
| `w_2ndWon` / `l_2ndWon` | Second serve points won |
| `w_SvGms` / `l_SvGms` | Service games played |
| `w_bpSaved` / `l_bpSaved` | Break points saved |
| `w_bpFaced` / `l_bpFaced` | Break points faced |
| `winner_rank` / `loser_rank` | ATP ranking at tournament start |
| `winner_rank_points` / `loser_rank_points` | Ranking points |

Match stats are integer totals (not percentages). Percentages can be derived (e.g., 1st serve % = `w_1stIn / w_svpt`). Stats coverage is generally 1991–present for tour-level; some matches have missing stats.

## Data Sources

**Jeff Sackmann / Tennis Abstract (2015–2024)**
Repository: [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
Attribution required. Non-commercial use only.
Cite as: Jeff Sackmann / Tennis Abstract, [tennisabstract.com](https://tennisabstract.com/)

**TML Database (2025–2026)**
Repository: [github.com/Tennismylife/TML-Database](https://github.com/Tennismylife/TML-Database)
Daily-updated ATP results, compatible column schema (with minor structural differences resolved at merge time).

## License

The Sackmann portion is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Any derivative work must attribute Jeff Sackmann / Tennis Abstract and remain non-commercial.
