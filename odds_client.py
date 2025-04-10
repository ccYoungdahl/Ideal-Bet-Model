import httpx
import logging

logger = logging.getLogger(__name__)

API_KEY = "37302c8fdd8c461d516159c28906cce2"
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

async def get_odds(event_id: str, bookmaker: str = None):
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "eventIds": event_id
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)

    if response.status_code != 200:
        logger.error(f"Odds API error: {response.status_code} {response.text}")
        raise RuntimeError(f"Odds API failed with status {response.status_code}: {response.text}")

    odds_data = response.json()

    if not odds_data:
        raise ValueError(f"No odds data returned for event_id: {event_id}")

    event = odds_data[0]

    spread_point = None
    total_point = None
    moneyline_home = None
    moneyline_away = None

    # Loop through bookmakers
    for bookmaker_data in event.get('bookmakers', []):
        if bookmaker and bookmaker_data['key'] != bookmaker:
            continue  # Skip unwanted bookmakers

        # Moneyline market (h2h)
        h2h_market = next((m for m in bookmaker_data['markets'] if m['key'] == 'h2h'), None)
        if h2h_market:
            for outcome in h2h_market.get('outcomes', []):
                if outcome['name'] == event['home_team']:
                    moneyline_home = outcome.get('price')
                elif outcome['name'] == event['away_team']:
                    moneyline_away = outcome.get('price')

        # Spread market
        spread_market = next((m for m in bookmaker_data['markets'] if m['key'] == 'spreads'), None)
        if spread_market:
            for outcome in spread_market.get('outcomes', []):
                if outcome['name'] == event['home_team']:
                    spread_point = outcome.get('point')

        # Totals market
        totals_market = next((m for m in bookmaker_data['markets'] if m['key'] == 'totals'), None)
        if totals_market:
            for outcome in totals_market.get('outcomes', []):
                if outcome['name'].lower() == 'over':
                    total_point = outcome.get('point')

        # Exit early if all found
        if all([spread_point, total_point, moneyline_home, moneyline_away]):
            break

    logger.info(f"Odds fetched for event {event_id}: Spread {spread_point}, Total {total_point}, Moneyline Home {moneyline_home}, Away {moneyline_away}")

    return {
        "spread_point": spread_point,
        "outcome_point_Over": total_point,
        "moneyline_home": moneyline_home,
        "moneyline_away": moneyline_away
    }
