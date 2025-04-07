# model_loader.py

import joblib
import pandas as pd

# Load models once at startup
moneyline_model = joblib.load("models/moneyline_model.pkl")
spread_model = joblib.load("models/spread_model.pkl")
overunder_model = joblib.load("models/overunder_model.pkl")

def get_model(market_type):
    if market_type == "moneyline":
        return moneyline_model
    elif market_type == "spread":
        return spread_model
    elif market_type == "overunder":
        return overunder_model
    else:
        raise ValueError("Unsupported market type")

def predict(features, model):
    X = pd.DataFrame([features])
    prob = model.predict_proba(X)[0][1]
    return prob
