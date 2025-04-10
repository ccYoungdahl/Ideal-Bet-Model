🏀 NBA Betting Model API
A FastAPI-powered machine learning API that delivers betting recommendations for NBA games based on live odds and cached team stats.

✅ Uses real-time odds from the Odds API
✅ Builds features dynamically from cached NBA game logs
✅ Supports moneyline, spread, and over/under markets
✅ Clean JSON API, production-ready, and fast

🚀 Features
Live Odds Integration — pulls real-time spread, total points, and moneyline odds at prediction time

Cached Team Stats — locally cached NBA team game logs for fast, reliable feature building

Automated Model Selection — routes requests to the correct ML model (moneyline, spread, or over/under)

User-Friendly Input — requires minimal frontend input (home_team, away_team, market, event_id, etc.)

Scalable & Async — optimized for cloud deployment (Render), fully async

📦 Project Structure
bash
Copy
Edit
/betting_model_api/
│
├── main.py                 # FastAPI API entrypoint
├── feature_builder.py      # Builds feature vectors from cached NBA data
├── odds_client.py          # Fetches live odds from Odds API
├── model_loader.py         # Loads ML models and predicts outcomes
├── utils.py                # Helper functions (probabilities, formatting, etc.)
├── models/                 # Saved ML models
│   ├── moneyline_model.pkl
│   ├── spread_model.pkl
│   └── overunder_model.pkl
├── team_game_logs.csv      # Cached NBA game logs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
⚙️ Setup
Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run locally
bash
Copy
Edit
uvicorn main:app --reload
Access the Swagger UI
arduino
Copy
Edit
http://127.0.0.1:8000/docs
🔌 API Usage
Endpoint
bash
Copy
Edit
POST /predict
Request Payload
json
Copy
Edit
{
  "home_team": "LAL",
  "away_team": "BOS",
  "user_team": "home",         // "home", "away", "over", "under"
  "market": "spread",          // "moneyline", "spread", "overunder"
  "event_id": "xyz123",        // Event ID from Odds API
  "bookmaker": "draftkings"    // Optional bookmaker (defaults to consensus)
}
Response
json
Copy
Edit
{
  "bet_type": "spread",
  "teams": {
    "home": "LAL",
    "away": "BOS"
  },
  "user_team": "home",
  "model_prob": 0.652,
  "value_edge": 0.087,
  "model_recommendation": "Bet",
  "confidence_level": "High"
}

