# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from feature_builder import build_feature_vector
from model_loader import get_model, predict
from utils import implied_probability, edge_level, to_py

app = FastAPI()

class UserBet(BaseModel):
    home_team: str
    away_team: str
    user_team: str  # "home", "away", "over", "under"
    user_odds: float
    market: str  # "moneyline", "spread", "overunder"

@app.post("/predict")
async def predict_bet(user_bet: UserBet):
    data = user_bet.dict()

    # Step 1: Build features
    features = await build_feature_vector(data['home_team'], data['away_team'])

    # Step 2: Add user odds info to features
    features['implied_prob_home'] = implied_probability(data['user_odds']) if data['user_team'] == 'home' else 1 - implied_probability(data['user_odds'])
    features['implied_prob_away'] = 1 - features['implied_prob_home']
    features['spread_point'] = -4.5  # Optional: enhance with Odds API
    features['outcome_point_Over'] = 227.5  # Optional: enhance with Odds API

    # Step 3: Get model and predict
    model = get_model(data['market'])
    model_prob_home = predict(features, model)

    # Step 4: Adjust model probability for user team
    if data['market'] == "overunder":
        model_prob = model_prob_home if data['user_team'] == "over" else 1 - model_prob_home
    else:
        model_prob = model_prob_home if data['user_team'] == "home" else 1 - model_prob_home

    # Step 5: Calculate edge and recommendation
    implied_prob = implied_probability(data["user_odds"])
    value_edge = model_prob - implied_prob
    recommendation = "Bet" if value_edge > 0.01 else "Pass"
    confidence = edge_level(value_edge)

    return {
        "bet_type": data['market'],
        "teams": {
            "home": data['home_team'],
            "away": data['away_team']
        },
        "user_team": data['user_team'],
        "user_odds": to_py(data['user_odds']),
        "implied_prob": to_py(round(implied_prob, 3)),
        "model_prob": to_py(round(model_prob, 3)),
        "value_edge": to_py(round(value_edge, 3)),
        "model_recommendation": recommendation,
        "confidence_level": confidence
    }

