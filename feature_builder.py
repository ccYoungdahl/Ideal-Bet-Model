# feature_builder.py

import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read cached game logs
try:
    game_logs_df = pd.read_csv('team_game_logs.csv')
    game_logs_df['GAME_DATE'] = pd.to_datetime(game_logs_df['GAME_DATE'])
    logger.info("✅ Loaded team game logs cache successfully.")
except FileNotFoundError:
    logger.error("❌ team_game_logs.csv not found. Run nba_cache_builder.py first!")
    raise

def process_team_stats(team_abbreviation):
    team_log_df = game_logs_df[game_logs_df['TEAM_ABBREVIATION'] == team_abbreviation]

    if team_log_df.empty:
        raise ValueError(f"No game logs found for team abbreviation '{team_abbreviation}'.")

    # Ensure correct order
    team_log_df = team_log_df.sort_values(by='GAME_DATE', ascending=False).reset_index(drop=True)

    if team_log_df.shape[0] < 3:
        raise ValueError(f"Insufficient game data for team '{team_abbreviation}' (min 3 games required).")

    team_log_df['REST_DAYS'] = team_log_df['GAME_DATE'].diff().dt.days.fillna(0).astype(int)
    team_log_df['WIN'] = team_log_df['WL'] == 'W'
    win_streak = team_log_df['WIN'].head(min(10, len(team_log_df))).sum()

    roll3 = team_log_df.head(3).mean(numeric_only=True)
    roll5 = team_log_df.head(min(5, len(team_log_df))).mean(numeric_only=True)

    last_game_rest = team_log_df['REST_DAYS'].iloc[0]
    is_back_to_back = int(last_game_rest <= 1)

    return {
        'pts_roll3': roll3['PTS'],
        'pts_roll5': roll5['PTS'],
        'rest_days': last_game_rest,
        'win_streak_last_10': win_streak,
        'fg_pct_roll3': roll3['FG_PCT'],
        'reb_roll3': roll3['REB'],
        'tov_roll3': roll3['TOV'],
        'is_back_to_back': is_back_to_back
    }

async def build_feature_vector(home_team_abbr, away_team_abbr):
    try:
        logger.info(f"Building feature vector from cache for {home_team_abbr} vs {away_team_abbr}")

        home_stats = process_team_stats(home_team_abbr)
        away_stats = process_team_stats(away_team_abbr)

        features = {
            'home_avg_pts_last_3': home_stats['pts_roll3'],
            'home_avg_pts_last_5': home_stats['pts_roll5'],
            'home_rest_days': home_stats['rest_days'],
            'home_win_streak_last_10': home_stats['win_streak_last_10'],
            'away_avg_pts_last_3': away_stats['pts_roll3'],
            'away_avg_pts_last_5': away_stats['pts_roll5'],
            'away_rest_days': away_stats['rest_days'],
            'away_win_streak_last_10': away_stats['win_streak_last_10'],
            'form_diff_pts_3': home_stats['pts_roll3'] - away_stats['pts_roll3'],
            'rest_advantage': home_stats['rest_days'] - away_stats['rest_days'],
            'win_streak_diff': home_stats['win_streak_last_10'] - away_stats['win_streak_last_10'],
            'fg_pct_diff_form': home_stats['fg_pct_roll3'] - away_stats['fg_pct_roll3'],
            'reb_diff_form': home_stats['reb_roll3'] - away_stats['reb_roll3'],
            'tov_diff_form': home_stats['tov_roll3'] - away_stats['tov_roll3'],
            'is_home_back_to_back': home_stats['is_back_to_back'],
            'is_away_back_to_back': away_stats['is_back_to_back'],
            'implied_prob_home': None,
            'implied_prob_away': None,
            'spread_point': None,
            'outcome_point_Over': None,
        }

        logger.info(f"✅ Feature vector built successfully from cache.")
        return features

    except Exception as e:
        logger.error(f"Error building feature vector from cache: {e}")
        raise

