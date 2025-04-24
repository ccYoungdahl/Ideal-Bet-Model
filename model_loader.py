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

# ---------- helper --------------------------------------------------
def _prepare_X(feat_dict: dict, feature_names: list[str]) -> pd.DataFrame:
    """Return DF with exactly feature_names; missing cols get 0.0."""
    return pd.DataFrame([{k: feat_dict.get(k, 0.0) for k in feature_names}])

# ---------- main API -----------------------------------------------
def predict_prob(features: dict, market: str, user_side: str) -> float:
    # ----- moneyline (binary) --------------------------------------
    if market == "moneyline":
        fns   = moneyline_model.get_booster().feature_names
        X     = _prepare_X(features, fns)
        p_home = moneyline_model.predict_proba(X)[0][1]
        return p_home if user_side == "home" else 1 - p_home

    # ----- spread (regression) -------------------------------------
    if market == "spread":
        fns = spread_reg.get_booster().feature_names
        X   = _prepare_X(features, fns)
        margin = spread_reg.predict(X)[0]            # + ⇒ home beats line
        p_home_cover = 1 - norm.cdf(-margin / SPREAD_SIGMA)
        return p_home_cover if user_side == "home" else 1 - p_home_cover

    # ----- over/under ---------------------------------------------
    if market == "overunder":
        fns = overunder_model.get_booster().feature_names
        X   = _prepare_X(features, fns)
        if hasattr(overunder_model, "predict_proba"):
            p_over = overunder_model.predict_proba(X)[0][1]
        else:
            margin = overunder_model.predict(X)[0]   # + ⇒ over
            p_over = 1 - norm.cdf(-margin / OU_SIGMA)
        return p_over if user_side == "over" else 1 - p_over

    raise ValueError(f"Unsupported market: {market}")
