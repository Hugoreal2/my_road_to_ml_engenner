# Portuguese Liga Soccer Match Predictor

This project fetches soccer match data from the Portuguese First League (Liga Portugal) using TheSportsDB API and prepares it for machine learning predictions.

## Project Structure

- `fetch_league_data.py` - Fetches match data from TheSportsDB API and saves to CSV
- `predict_matches.py` - Machine learning model for match prediction
- `requirements.txt` - Python dependencies
- `portugal_liga_*.csv` - Generated CSV files with match data
- `venv/` - Virtual environment (created automatically)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Run the data fetcher:
```bash
python fetch_league_data.py
```

4. Run the prediction model (once you have enough data):
```bash
python predict_matches.py
```

## API Information

- **API**: TheSportsDB (https://www.thesportsdb.com)
- **Free API Key**: `3`
- **Rate Limit**: 30 requests per minute (free tier)
- **League ID**: 4344 (Portuguese Liga)

## Data Fields

The CSV file includes the following fields for each match:

- `match_id` - Unique match identifier
- `date` - Match date
- `time` - Match time
- `season` - Season (e.g., "2023-2024")
- `round` - Round number
- `venue` - Stadium name
- `home_team` / `away_team` - Team names
- `home_score` / `away_score` - Final scores
- `home_goals_halftime` / `away_goals_halftime` - Halftime scores
- `home_shots` / `away_shots` - Shot statistics
- `spectators` - Attendance
- `match_finished` - Whether the match is completed

## Next Steps

After fetching the data, you can:

1. Perform exploratory data analysis (EDA)
2. Engineer features (home advantage, recent form, head-to-head, etc.)
3. Build prediction models (Logistic Regression, Random Forest, XGBoost, etc.)
4. Evaluate model performance

## Upgrade to Premium

For more features and data:
- Premium API key: $9/month
- Benefits: More requests, livescores, video highlights, V2 API access
- Sign up at: https://www.thesportsdb.com/pricing
