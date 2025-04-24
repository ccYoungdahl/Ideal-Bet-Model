# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from feature_builder import build_feature_vector
from model_loader import get_model, predict
from odds_client import get_odds
from utils import implied_probability, edge_level, to_py
from typing import Optional
import logging
import numpy as np
from feature_builder import LEAGUE_PACE

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request schema
class UserBet(BaseModel):
    home_team: str
    away_team: str
    user_team: str  # "home", "away", "over", "under"
    market: str     # "moneyline", "spread", "overunder"
    event_id: str
    bookmaker: Optional[str] = None  # Optional bookmaker

@app.post("/predict")
async def predict_bet(user_bet: UserBet):
    data = user_bet.dict()
    logger.info(f"🚀 Received prediction request: {data}")

    # Step 1: Build team features (cached)
    features = await build_feature_vector(data['home_team'], data['away_team'])

    # Step 2: Fetch live odds
    odds_data = await get_odds(event_id=data['event_id'], bookmaker=data.get('bookmaker'))
    features['spread_point'] = odds_data['spread_point']
    features['outcome_point_Over'] = odds_data['outcome_point_Over']
    #Calc Pace-Adjusted Total
    if features["outcome_point_Over"] is not None:
        avg_pace = np.nanmean([features["home_pace_last5"], features["away_pace_last5"]])
        if not np.isnan(avg_pace):
            features["pace_adj_total"] = features["outcome_point_Over"] * (avg_pace / LEAGUE_PACE)

    # Step 3: Calculate implied probabilities from moneyline odds
    if not odds_data['moneyline_home'] or not odds_data['moneyline_away']:
        raise ValueError("Missing moneyline odds from Odds API")

    home_implied = implied_probability(odds_data['moneyline_home'])
    away_implied = implied_probability(odds_data['moneyline_away'])

    features['implied_prob_home'] = home_implied
    features['implied_prob_away'] = away_implied

   # ------------------------------------------------------------
    # 4. Get the correct model and probability for the user's side
    # ------------------------------------------------------------
    model = get_model(data['market'])
    model_prob_home = predict(features, model)          # P(home wins)  or  P(over)

    is_over_market = data['market'] == "overunder"
    user_is_fav    = (
        (is_over_market and data['user_team'] == "over") or
        (not is_over_market and data['user_team'] == "home")
    )
    model_prob = model_prob_home if user_is_fav else 1 - model_prob_home

    # ------------------------------------------------------------
    # 5. Model certainty  (distance from 0.50)
    # ------------------------------------------------------------
    certainty = abs(model_prob - 0.50)                 # 0–0.50 scale
    from utils import certainty_level, edge_level
    certainty_bucket = certainty_level(certainty)

    # ------------------------------------------------------------
    # 6. Edge versus bookmaker price
    # ------------------------------------------------------------
    if data['user_team'] in ["home", "away"]:
        implied_prob = home_implied if data['user_team'] == "home" else away_implied
    else:  # over / under
        implied_prob = (
            implied_probability(odds_data['over_odds'])
            if data['user_team'] == "over"
            else implied_probability(odds_data['under_odds'])
        )

    value_edge = model_prob - implied_prob

    # ------------------------------------------------------------
    # 7. Confidence label – take the worse of edge-bucket & certainty-bucket
    # ------------------------------------------------------------
    edge_bucket = edge_level(abs(value_edge))
    ordered = ["Negligible", "Low", "Moderate", "High"]        
    confidence = min(edge_bucket, certainty_bucket, key=ordered.index)

    # ------------------------------------------------------------
    # 8. Recommendation – only bet if edge ≥ 1 pp AND certainty ≥ Low
    # ------------------------------------------------------------
    recommendation = (
        "Bet" if value_edge > 0.01 and certainty >= 0.10 else "Pass"
    )

    # ------------------------------------------------------------
    # 9. Build the response  (unchanged keys)
    # ------------------------------------------------------------
    response = {
        "bet_type": data['market'],
        "teams": {"home": data['home_team'], "away": data['away_team']},
        "user_team": data['user_team'],
        "model_prob": to_py(round(model_prob, 3)),
        "value_edge": to_py(round(value_edge, 3)),
        "model_recommendation": recommendation,
        "confidence_level": confidence
    }
    logger.info(f"✅ Prediction completed: {response}")
    return response

