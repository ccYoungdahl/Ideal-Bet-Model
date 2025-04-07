# nba_cache_builder.py

import pandas as pd
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from datetime import datetime
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_team_game_logs(season='2024-25'):
    all_teams = teams.get_teams()
    all_logs = []

    for team in all_teams:
        try:
            team_name = team['full_name']
            team_abbr = team['abbreviation']
            team_id = team['id']

            logger.info(f"Fetching game log for {team_name} ({team_abbr})")

            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
            game_log['TEAM_ID'] = team_id
            game_log['TEAM_ABBREVIATION'] = team_abbr
            game_log['TEAM_NAME'] = team_name

            all_logs.append(game_log)

            # Pause to avoid rate limiting
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error fetching {team_name}: {e}")
            continue

    if all_logs:
        df_all = pd.concat(all_logs, ignore_index=True)
        df_all.to_csv('team_game_logs.csv', index=False)
        logger.info("✅ Cache built successfully: team_game_logs.csv")
    else:
        logger.warning("No data fetched.")

if __name__ == "__main__":
    fetch_team_game_logs()
