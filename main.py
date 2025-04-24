# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import logging, numpy as np

from feature_builder import build_feature_vector, LEAGUE_PACE
from model_loader import predict_prob
from odds_client import get_odds
from utils import (
    implied_probability, edge_level, certainty_level, to_py,
)

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
class UserBet(BaseModel):
    home_team: str
    away_team: str
    user_team: str   # "home", "away", "over", "under"
    market: str      # "moneyline", "spread", "overunder"
    event_id: str
    bookmaker: Optional[str] = None
# -----------------------------------------------------------

@app.post("/predict")
async def predict_bet(user_bet: UserBet):
    data = user_bet.dict()
    logger.info(f"🚀 Received prediction request: {data}")

    # 1) team features --------------------------------------
    features = await build_feature_vector(data["home_team"], data["away_team"])

    # 2) live odds ------------------------------------------
    odds_data = await get_odds(event_id=data["event_id"], bookmaker=data.get("bookmaker"))
    features["spread_point"]        = odds_data["spread_point"]
    features["outcome_point_Over"]  = odds_data["outcome_point_Over"]

    # pace-adjusted total
    if features["outcome_point_Over"] is not None:
        avg_pace = np.nanmean([features["home_pace_last5"], features["away_pace_last5"]])
        if not np.isnan(avg_pace):
            features["pace_adj_total"] = features["outcome_point_Over"] * (avg_pace / LEAGUE_PACE)

    # 3) implied probs from moneyline -----------------------
    if not odds_data["moneyline_home"] or not odds_data["moneyline_away"]:
        raise ValueError("Missing moneyline odds from Odds API")

    home_implied = implied_probability(odds_data["moneyline_home"])
    away_implied = implied_probability(odds_data["moneyline_away"])
    features["implied_prob_home"] = home_implied
    features["implied_prob_away"] = away_implied

    # 4) model probability for the user's side --------------
    model_prob = predict_prob(features, data["market"], data["user_team"])

    # 5) certainty ------------------------------------------
    certainty       = abs(model_prob - 0.5)
    certainty_bucket = certainty_level(certainty)

    
    # step 6 – implied probability for this bet  (vig-free)--------------
    if market in ("moneyline", "spread"):
        overround = home_raw + away_raw
        fair_home = home_raw / overround
        fair_away = away_raw / overround
        implied_prob = fair_home if user_team=="home" else fair_away
    else:  # over/under
        overround_tot = over_raw + under_raw
        implied_prob = (over_raw  / overround_tot if user_team=="over"
                        else under_raw / overround_tot)


    value_edge = model_prob - implied_prob

    # 7) confidence label -----------------------------------
    edge_bucket = edge_level(abs(value_edge))
    order = ["Negligible", "Low", "Moderate", "High"]
    confidence = min(edge_bucket, certainty_bucket, key=order.index)

    # 8) bet / pass rule ------------------------------------
    recommendation = "Bet" if value_edge > 0.01 and certainty >= 0.10 else "Pass"

    # 9) response -------------------------------------------
    response = {
        "bet_type": data["market"],
        "teams": {"home": data["home_team"], "away": data["away_team"]},
        "user_team": data["user_team"],
        "model_prob": to_py(round(model_prob, 3)),
        "value_edge": to_py(round(value_edge, 3)),
        "model_recommendation": recommendation,
        "confidence_level": confidence,
    }
    logger.info(f"✅ Prediction completed: {response}")
    return response


