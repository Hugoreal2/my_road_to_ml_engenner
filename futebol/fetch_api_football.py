"""
API-Football - Portuguese League Data Fetcher
FREE TIER: 100 requests/day (much better than TheSportsDB!)

SETUP:
1. Register FREE at: https://dashboard.api-football.com/register
2. Get your API key from: https://dashboard.api-football.com/profile?access
3. Paste it below in API_KEY

NO CREDIT CARD REQUIRED!
"""

import requests
import pandas as pd
from datetime import datetime
import time

# API Configuration
BASE_URL = "https://v3.football.api-sports.io"
API_KEY = "2cfe7afa0950e067cd3997299e334350"

# League Configuration
LEAGUE_ID = 94  # Portuguese Primeira Liga
SEASONS = ["2022", "2023", "2024", "2025"]  # Free API: 2022-2024 only (2025 = current season)

class APIFootballFetcher:
    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
        self.base_url = BASE_URL
        self.headers = {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': api_key
        }
        self.request_count = 0
        
    def make_request(self, endpoint, params=None):
        """Make API request with rate limiting"""
        url = f"{self.base_url}/{endpoint}"
        self.request_count += 1
        print(f"  [Request #{self.request_count}] {endpoint}...")
        
        response = requests.get(url, headers=self.headers, params=params)
        time.sleep(1)  # Rate limiting - be nice to the API
        
        if response.status_code == 200:
            data = response.json()
            if data.get('errors') and len(data['errors']) > 0:
                print(f"  ✗ API Error: {data['errors']}")
                return None
            return data.get('response', [])
        else:
            print(f"  ✗ HTTP Error: {response.status_code}")
            return None
    
    def get_fixtures(self, league_id, season):
        """Get all fixtures for a league and season"""
        params = {
            'league': league_id,
            'season': season
        }
        print(f"\nFetching fixtures for season {season}...")
        return self.make_request('fixtures', params)

def flatten_fixture_data(fixtures):
    """Convert fixtures to flat structure for CSV"""
    matches = []
    
    for fixture in fixtures:
        if fixture is None:
            continue
        
        match = {
            'match_id': fixture['fixture']['id'],
            'date': fixture['fixture']['date'],
            'timestamp': fixture['fixture']['timestamp'],
            'season': fixture['league']['season'],
            'round': fixture['league']['round'],
            'venue': fixture['fixture']['venue']['name'] if fixture['fixture']['venue'] else None,
            'city': fixture['fixture']['venue']['city'] if fixture['fixture']['venue'] else None,
            'referee': fixture['fixture']['referee'],
            'status': fixture['fixture']['status']['long'],
            'status_short': fixture['fixture']['status']['short'],
            'league': fixture['league']['name'],
            'country': fixture['league']['country'],
            'home_team': fixture['teams']['home']['name'],
            'away_team': fixture['teams']['away']['name'],
            'home_team_id': fixture['teams']['home']['id'],
            'away_team_id': fixture['teams']['away']['id'],
            'home_score': fixture['goals']['home'],
            'away_score': fixture['goals']['away'],
            'home_goals_halftime': fixture['score']['halftime']['home'],
            'away_goals_halftime': fixture['score']['halftime']['away'],
            'home_goals_fulltime': fixture['score']['fulltime']['home'],
            'away_goals_fulltime': fixture['score']['fulltime']['away'],
            'match_finished': fixture['fixture']['status']['short'] == 'FT',
        }
        matches.append(match)
    
    return matches

def main():
    """Main function"""
    
    print("=" * 70)
    print("API-FOOTBALL - PORTUGUESE LIGA DATA FETCHER")
    print("=" * 70)
    print(f"League ID: {LEAGUE_ID}")
    print(f"Seasons: {SEASONS}")
    print(f"API Key: {'✓ Configured' if API_KEY != 'YOUR_API_KEY_HERE' else '✗ NOT SET'}")
    print()
    print("FREE TIER: 100 requests/day (no credit card required)")
    print("Get your FREE key: https://dashboard.api-football.com/register")
    print("=" * 70)
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ ERROR: Please set your API key!")
        print("\n1. Register FREE at: https://dashboard.api-football.com/register")
        print("2. Get your key from: https://dashboard.api-football.com/profile?access")
        print("3. Paste it in this file at line 15: API_KEY = 'your_key_here'")
        print("\nNo credit card needed! 100% FREE!")
        return
    
    fetcher = APIFootballFetcher()
    all_matches = []
    
    # Fetch data for each season
    for season in SEASONS:
        fixtures = fetcher.get_fixtures(LEAGUE_ID, season)
        
        if fixtures:
            matches = flatten_fixture_data(fixtures)
            all_matches.extend(matches)
            print(f"  ✓ Season {season}: {len(matches)} matches")
        else:
            print(f"  ✗ Season {season}: No data")
    
    print(f"\n{'=' * 70}")
    print(f"Total API requests made: {fetcher.request_count}")
    print(f"Remaining today: ~{100 - fetcher.request_count} (free tier: 100/day)")
    print(f"{'=' * 70}")
    
    # Save to CSV
    if all_matches:
        df = pd.DataFrame(all_matches)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        filename = f"portugal_liga_api_football_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n✓ Successfully saved {len(df)} matches to '{filename}'")
        print(f"\nDataset Info:")
        print(f"  - Total matches: {len(df)}")
        print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  - Seasons: {df['season'].unique().tolist()}")
        print(f"  - Columns: {len(df.columns)}")
        
        print("\nSample data (first 3 rows):")
        print(df[['date', 'home_team', 'away_team', 'home_score', 'away_score']].head(3))
        
        finished_matches = df[df['match_finished'] == True]
        print(f"\nMatch Statistics:")
        print(f"  - Finished matches: {len(finished_matches)}")
        print(f"  - Pending matches: {len(df) - len(finished_matches)}")
        
        if len(finished_matches) > 0:
            print(f"  - Average home goals: {finished_matches['home_score'].mean():.2f}")
            print(f"  - Average away goals: {finished_matches['away_score'].mean():.2f}")
    else:
        print("\n✗ No match data retrieved")
    
    print("\n" + "=" * 70)
    print("Done! With free API you get ALL matches (1500+), not just 15!")
    print("=" * 70)

if __name__ == "__main__":
    main()
