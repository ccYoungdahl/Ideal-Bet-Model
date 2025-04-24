# model_loader.py
import joblib, json
from pathlib import Path
import pandas as pd
from scipy.stats import norm

MODELS = Path("models")

# ---- load models ---------------------------------------------------
moneyline_model  = joblib.load(MODELS / "moneyline_model.pkl")
spread_reg       = joblib.load(MODELS / "spread_model_reg.pkl")
overunder_model  = joblib.load(MODELS / "overunder_model_v2.pkl")

with open(MODELS / "spread_sigma.json") as f:
    SPREAD_SIGMA = json.load(f)["sigma"]          # ≈ 6.56

OU_SIGMA = 11.0                                   # keep / tweak later

# --------------------------------------------------------------------
def predict_prob(features: dict, market: str, user_side: str) -> float:
    """
    Return probability user's side wins.
      • moneyline : classifier
      • spread    : regression → prob cover
      • overunder : classifier OR regression
    """
    X = pd.DataFrame([features])

    # --- moneyline --------------------------------------------------
    if market == "moneyline":
        p_home = moneyline_model.predict_proba(X)[0][1]
        return p_home if user_side == "home" else 1 - p_home

    # --- spread (reg) ----------------------------------------------
    if market == "spread":
        margin = spread_reg.predict(X)[0]              # + ⇒ home beats line
        p_cover_home = 1 - norm.cdf(-margin / SPREAD_SIGMA)
        return p_cover_home if user_side == "home" else 1 - p_cover_home

    # --- over/under -------------------------------------------------
    if market == "overunder":
        if hasattr(overunder_model, "predict_proba"):
            p_over = overunder_model.predict_proba(X)[0][1]
        else:                                           # regression margin
            margin = overunder_model.predict(X)[0]      # + ⇒ game over
            p_over = 1 - norm.cdf(-margin / OU_SIGMA)
        return p_over if user_side == "over" else 1 - p_over

    raise ValueError(f"Unsupported market: {market}")

