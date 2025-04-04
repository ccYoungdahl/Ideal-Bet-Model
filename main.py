# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from features import common_features

app = FastAPI()

# === Load models ===
moneyline_model = joblib.load("models/moneyline_model.pkl")
spread_model = joblib.load("models/spread_model.pkl")
over_model = joblib.load("models/overunder_model.pkl")

# === Define input schema ===
class GameInput(BaseModel):
    home_avg_pts_last_3: float
    home_avg_pts_last_5: float
    home_rest_days: float
    home_win_streak_last_10: float
    away_avg_pts_last_3: float
    away_avg_pts_last_5: float
    away_rest_days: float
    away_win_streak_last_10: float
    form_diff_pts_3: float
    rest_advantage: float
    win_streak_diff: float
    fg_pct_diff_form: float
    reb_diff_form: float
    tov_diff_form: float
    is_home_back_to_back: int
    is_away_back_to_back: int
    implied_prob_home: float
    implied_prob_away: float
    spread_point: float
    outcome_point_Over: float
    user_odds: float
    home_team: str
    away_team: str
    user_team: str  # "home" or "away"

# === Utility functions ===
def implied_probability(odds):
    odds = float(odds)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def edge_level(edge):
    if edge >= 0.1:
        return "High"
    elif edge >= 0.05:
        return "Moderate"
    elif edge >= 0.01:
        return "Low"
    else:
        return "Negligible"

def build_response(input_data, model_prob_raw, bet_type):
    user_team = input_data["user_team"]
    user_odds = input_data["user_odds"]
    implied_prob = implied_probability(user_odds)

    # Adjust model probability to reflect user's team
    model_prob = model_prob_raw if user_team == "home" else 1 - model_prob_raw

    value_edge = model_prob - implied_prob
    recommendation = "Bet" if value_edge > 0.01 else "Pass"
    confidence = edge_level(value_edge)

    return {
        "bet_type": bet_type,
        "teams": {
            "home": input_data["home_team"],
            "away": input_data["away_team"]
        },
        "user_team": user_team,
        "user_odds": user_odds,
        "implied_prob": round(implied_prob, 3),
        "model_prob": round(model_prob, 3),
        "value_edge": round(value_edge, 3),
        "model_recommendation": recommendation,
        "confidence_level": confidence
    }

# === API Endpoints ===

@app.post("/predict/moneyline")
def predict_moneyline(input_data: GameInput):
    data = input_data.dict()
    X = pd.DataFrame([data])[common_features]
    model_prob_home = moneyline_model.predict_proba(X)[0][1]

    return build_response(data, model_prob_home, "moneyline")

@app.post("/predict/spread")
def predict_spread(input_data: GameInput):
    data = input_data.dict()
    X = pd.DataFrame([data])[common_features]
    model_prob_home_cover = spread_model.predict_proba(X)[0][1]

    return build_response(data, model_prob_home_cover, "spread")

@app.post("/predict/overunder")
def predict_overunder(input_data: GameInput):
    data = input_data.dict()
    X = pd.DataFrame([data])[common_features]
    model_prob_over = over_model.predict_proba(X)[0][1]

    # For over/under, "user_team" = "over" or "under"
    user_team = data["user_team"]
    model_prob = model_prob_over if user_team == "over" else 1 - model_prob_over

    value_edge = model_prob - implied_probability(data["user_odds"])
    recommendation = "Bet" if value_edge > 0.01 else "Pass"
    confidence = edge_level(value_edge)

    return {
        "bet_type": "overunder",
        "teams": {
            "home": data["home_team"],
            "away": data["away_team"]
        },
        "user_team": user_team,
        "user_odds": data["user_odds"],
        "implied_prob": round(implied_probability(data["user_odds"]), 3),
        "model_prob": round(model_prob, 3),
        "value_edge": round(value_edge, 3),
        "model_recommendation": recommendation,
        "confidence_level": confidence
    }