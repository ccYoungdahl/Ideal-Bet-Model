# feature_builder.py

import pandas as pd
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from datetime import datetime

# Helper: Get NBA team ID
def get_team_id(team_abbreviation):
    all_teams = teams.get_teams()
    team = next((t for t in all_teams if t['abbreviation'] == team_abbreviation), None)
    if not team:
        raise ValueError(f"Team abbreviation '{team_abbreviation}' not found.")
    return team['id']

# Helper: Process game log into features
def process_team_stats(game_log_df):
    # Ensure GAME_DATE is datetime
    game_log_df['GAME_DATE'] = pd.to_datetime(game_log_df['GAME_DATE'])

    #  Sort by date descending (latest games first)
    game_log_df = game_log_df.sort_values(by='GAME_DATE', ascending=False).reset_index(drop=True)

    # Calculate rest days
    game_log_df['REST_DAYS'] = game_log_df['GAME_DATE'].diff().dt.days.fillna(0).astype(int)

    # Win streak over last 10 games
    game_log_df['WIN'] = game_log_df['WL'] == 'W'
    win_streak = game_log_df['WIN'].head(10).sum()

    # Rolling averages
    roll3 = game_log_df.head(3).mean(numeric_only=True)
    roll5 = game_log_df.head(5).mean(numeric_only=True)

    # Last game rest info
    last_game_rest = game_log_df['REST_DAYS'].iloc[0]
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


# Main function: Build feature vector for game
async def build_feature_vector(home_team_abbr, away_team_abbr):
    # Fetch game logs
    home_team_id = get_team_id(home_team_abbr)
    away_team_id = get_team_id(away_team_abbr)

    home_log = teamgamelog.TeamGameLog(team_id=home_team_id, season="2024-25").get_data_frames()[0]
    away_log = teamgamelog.TeamGameLog(team_id=away_team_id, season="2024-25").get_data_frames()[0]

    # Process stats
    home_stats = process_team_stats(home_log)
    away_stats = process_team_stats(away_log)

    # Build features
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
        # Optional odds-based features will be filled in main.py
        'implied_prob_home': None,
        'implied_prob_away': None,
        'spread_point': None,
        'outcome_point_Over': None,
    }

    return features
