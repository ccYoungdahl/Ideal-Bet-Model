# -*- coding: utf-8 -*-
COMMON_BASE = [
    "home_win_pct_last10","away_win_pct_last10","win_pct_diff_last10",
    "rest_advantage","win_streak_diff","home_rest_days","away_rest_days","is_home_back_to_back",
    "is_away_back_to_back","fg_pct_roll3_home","fg_pct_roll3_away","home_fg3a_last5","away_fg3a_last5","fg3a_diff_last5",
    "home_fta_last5","away_fta_last5","fta_diff_last5",
]

MONEYLINE_FEATURES = COMMON_BASE + [
    "home_offrtg_last5","away_offrtg_last5","offrtg_diff_last5",
    "implied_prob_home","implied_prob_away","tov_diff_form","home_win_pct_season","away_win_pct_season",
    "win_pct_diff_season"
]

SPREAD_FEATURES = COMMON_BASE + [
    "pace_diff_last5","netrtg_diff_last5","spread_point",
    "efg_pct_diff_last5","fga_diff_last5","tov_diff_form","implied_prob_home","implied_prob_away","home_win_pct_season","away_win_pct_season",
    "win_pct_diff_season"
]

OVERUNDER_FEATURES = [
    "pace_adj_total","home_pace_last5","away_pace_last5","pace_diff_last5",
    "home_ppg_last5","away_ppg_last5","ppg_diff_last5",
    "offrtg_diff_last5","defrtg_diff_last5","netrtg_diff_last5","outcome_point_Over",
]


