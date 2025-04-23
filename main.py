# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from feature_builder import build_feature_vector
from model_loader import get_model, predict
from odds_client import get_odds
from utils import implied_probability, edge_level, to_py
from typing import Optional
import logging

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

    # Step 3: Calculate implied probabilities from moneyline odds
    if not odds_data['moneyline_home'] or not odds_data['moneyline_away']:
        raise ValueError("Missing moneyline odds from Odds API")

    home_implied = implied_probability(odds_data['moneyline_home'])
    away_implied = implied_probability(odds_data['moneyline_away'])

    features['implied_prob_home'] = home_implied
    features['implied_prob_away'] = away_implied

    # Step 4: Get model and predict
    model = get_model(data['market'])
    model_prob_home = predict(features, model)

    # Step 5: Adjust probability for user bet direction
    if data['market'] == "overunder":
        model_prob = model_prob_home if data['user_team'] == "over" else 1 - model_prob_home
    else:
        model_prob = model_prob_home if data['user_team'] == "home" else 1 - model_prob_home

    # Step 6: Calculate value edge
    if data['user_team'] in ["home", "away"]:
        implied_prob = home_implied if data['user_team'] == "home" else away_implied
    elif data['user_team'] in ["over", "under"]:
        if odds_data['over_odds'] is None or odds_data['under_odds'] is None:
            raise ValueError("Missing totals odds for Over/Under market")
        implied_prob = (
            implied_probability(odds_data['over_odds'])
            if data['user_team'] =="over"
            else implied_probability(odds_data['under_odds'])
        )
    else:
        raise ValueError("Invalid user_team value")

    value_edge = model_prob - implied_prob if implied_prob is not None else None
    recommendation = "Bet" if value_edge and value_edge > 0.01 else "Pass"
    confidence = edge_level(value_edge) if value_edge is not None else "Unknown"

    response = {
        "bet_type": data['market'],
        "teams": {
            "home": data['home_team'],
            "away": data['away_team']
        },
        "user_team": data['user_team'],
        "model_prob": to_py(round(model_prob, 3)),
        "value_edge": to_py(round(value_edge, 3)) if value_edge is not None else None,
        "model_recommendation": recommendation,
        "confidence_level": confidence
    }

    logger.info(f"✅ Prediction completed: {response}")
    return response

