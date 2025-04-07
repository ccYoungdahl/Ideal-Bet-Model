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

# === Input schema ===
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
    implied_prob_over: float  # ✅ New for Over/Under
    implied_prob_under: float  # ✅ New for Over/Under
    spread_point: float
    outcome_point_Over: float
    home_team: str
    away_team: str
    user_team: str  # "home", "away", "over", "under"

# === Utility functions ===

def edge_level(edge):
    if edge >= 0.1:
        return "High"
    elif edge >= 0.05:
        return "Moderate"
    elif edge >= 0.01:
        return "Low"
    else:
        return "Negligible"

def to_py(obj):
    """Ensure compatibility with FastAPI JSON response (Python native types)."""
    if hasattr(obj, "item"):
        return obj.item()
    return obj

def get_user_implied_prob(input_data, bet_type):
    user_team = input_data["user_team"]

    if bet_type in ["moneyline", "spread"]:
        return input_data["implied_prob_home"] if user_team == "home" else input_data["implied_prob_away"]
    elif bet_type == "overunder":
        return input_data["implied_prob_over"] if user_team == "over" else input_data["implied_prob_under"]
    
    return None

def build_response(input_data, model_prob_raw, bet_type):
    user_team = input_data["user_team"]
    user_implied_prob = get_user_implied_prob(input_data, bet_type)

    # Adjust model probability based on user bet direction
    if user_team in ["home", "over"]:
        model_prob = model_prob_raw
    else:
        model_prob = 1 - model_prob_raw

    # Compute value edge
    if user_implied_prob is not None:
        value_edge = model_prob - user_implied_prob
        recommendation = "Bet" if value_edge > 0.01 else "Pass"
        confidence = edge_level(value_edge)
    else:
        value_edge = None
        recommendation = "TBD"
        confidence = "Unknown"

    return {
        "bet_type": bet_type,
        "teams": {
            "home": input_data["home_team"],
            "away": input_data["away_team"]
        },
        "user_team": user_team,
        "implied_prob": to_py(round(user_implied_prob, 3)) if user_implied_prob is not None else None,
        "model_prob": to_py(round(model_prob, 3)),
        "value_edge": to_py(round(value_edge, 3)) if value_edge is not None else None,
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

    return build_response(data, model_prob_over, "overunder")
