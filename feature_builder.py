# feature_builder.py
import pandas as pd, numpy as np, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- load cached logs -----------------------------------------
logs = pd.read_csv("team_game_logs.csv", parse_dates=["GAME_DATE"])

# eFG% fallback if missing
if "EFG_PCT" not in logs.columns:
    logs["EFG_PCT"] = logs["FG_PCT"]

def possessions(r):
    return r["FGA"] + 0.44 * r["FTA"] - r["OREB"] + r["TOV"]

logs["POSS"]    = logs.apply(possessions, axis=1)
logs["PACE"]    = logs["POSS"]
logs["OFF_RTG"] = 100 * logs["PTS"] / logs["POSS"]

# opponent PTS for DEF_RTG
if "GAME_ID" in logs.columns:
    opp_pts = logs.groupby("GAME_ID")["PTS"].apply(list).to_dict()
    logs["DEF_RTG"] = logs.apply(lambda r: 100 * opp_pts[r["GAME_ID"]][1] / r["POSS"], axis=1)
elif "PTS_OPP" in logs.columns:
    logs["DEF_RTG"] = 100 * logs["PTS_OPP"] / logs["POSS"]
else:
    logs["DEF_RTG"] = 100 * logs["PTS"].mean() / logs["POSS"]

logs["NET_RTG"] = logs["OFF_RTG"] - logs["DEF_RTG"]
LEAGUE_PACE = logs["PACE"].mean()

# -------- helpers --------------------------------------------------
def _window(team, asof, n):
    g = logs[(logs.TEAM_ABBREVIATION == team) & (logs.GAME_DATE < asof)]
    return g.sort_values("GAME_DATE", ascending=False).head(n)

def roll(team, asof, col, n, default=np.nan):
    g = _window(team, asof, n)
    return g[col].mean() if len(g) else default

def win_pct(team, asof, n=None):
    g = _window(team, asof, n) if n else logs[(logs.TEAM_ABBREVIATION == team) &
                                              (logs.GAME_DATE < asof)]
    return (g["WL"] == "W").mean() if len(g) else 0.5

# -------------------------------------------------------------------
async def build_feature_vector(home, away):
    asof = datetime.utcnow()
    h, a = home, away

    # ---------- win % and rest -------------------------------------
    features = {
        "home_win_pct_last10": win_pct(h, asof, 10),
        "away_win_pct_last10": win_pct(a, asof, 10),
        "win_pct_diff_last10": win_pct(h, asof, 10) - win_pct(a, asof, 10),

        "home_win_pct_season": win_pct(h, asof),
        "away_win_pct_season": win_pct(a, asof),
        "win_pct_diff_season": win_pct(h, asof) - win_pct(a, asof),

        "home_rest_days": roll(h, asof, "REST_DAYS", 1, 0),
        "away_rest_days": roll(a, asof, "REST_DAYS", 1, 0),
        "is_home_back_to_back": int(roll(h, asof, "REST_DAYS", 1, 2) <= 1),
        "is_away_back_to_back": int(roll(a, asof, "REST_DAYS", 1, 2) <= 1),
    }
    features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

    # ---------- win streak diff ------------------------------------
    features["win_streak_diff"] = (
        roll(h, asof, "WIN", 10, 0) - roll(a, asof, "WIN", 10, 0)
    )

    # ---------- rolling FG% (3-game) -------------------------------
    features["fg_pct_roll3_home"] = roll(h, asof, "FG_PCT", 3, 0)
    features["fg_pct_roll3_away"] = roll(a, asof, "FG_PCT", 3, 0)

    # ---------- pace & ratings (5-game) ----------------------------
    features["home_pace_last5"]   = roll(h, asof, "PACE",    5, LEAGUE_PACE)
    features["away_pace_last5"]   = roll(a, asof, "PACE",    5, LEAGUE_PACE)
    features["pace_diff_last5"]   = features["home_pace_last5"] - features["away_pace_last5"]

    features["home_offrtg_last5"] = roll(h, asof, "OFF_RTG", 5)
    features["away_offrtg_last5"] = roll(a, asof, "OFF_RTG", 5)
    features["offrtg_diff_last5"] = features["home_offrtg_last5"] - features["away_offrtg_last5"]

    features["home_defrtg_last5"] = roll(h, asof, "DEF_RTG", 5)
    features["away_defrtg_last5"] = roll(a, asof, "DEF_RTG", 5)
    features["defrtg_diff_last5"] = features["home_defrtg_last5"] - features["away_defrtg_last5"]

    features["netrtg_diff_last5"] = (
        (features["home_offrtg_last5"] - features["home_defrtg_last5"]) -
        (features["away_offrtg_last5"] - features["away_defrtg_last5"])
    )

    # ---------- eFG% diff (5-game) ---------------------------------
    features["home_efg_pct_last5"] = roll(h, asof, "EFG_PCT", 5)
    features["away_efg_pct_last5"] = roll(a, asof, "EFG_PCT", 5)
    features["efg_pct_diff_last5"] = features["home_efg_pct_last5"] - features["away_efg_pct_last5"]

    # ---------- volume stats (5-game) ------------------------------
    for stat, prefix in [("PTS", "ppg"), ("FGA", "fga"), ("FG3A", "fg3a"), ("FTA", "fta")]:
        features[f"home_{prefix}_last5"] = roll(h, asof, stat, 5)
        features[f"away_{prefix}_last5"] = roll(a, asof, stat, 5)
        features[f"{prefix}_diff_last5"] = features[f"home_{prefix}_last5"] - features[f"away_{prefix}_last5"]

    # ---------- turnovers diff (3-game) ----------------------------
    features["tov_diff_form"] = roll(h, asof, "TOV", 3, 0) - roll(a, asof, "TOV", 3, 0)

    # ---------- placeholders filled later --------------------------
    features.update({
        "implied_prob_home": None,
        "implied_prob_away": None,
        "spread_point": None,
        "outcome_point_Over": None,
        "pace_adj_total": None,
    })

    logger.info(f"✅ Features built for {h} vs {a}")
    return features
