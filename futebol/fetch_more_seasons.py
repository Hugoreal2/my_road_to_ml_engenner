"""
Fetch additional seasons from API-Football and append to existing CSV
Uses the same API key and format as before
"""

import requests
import pandas as pd
import time
from datetime import datetime

API_KEY = "2cfe7afa0950e067cd3997299e334350"
BASE_URL = "https://v3.football.api-sports.io"
LEAGUE_ID = 94  # Portuguese Primeira Liga
SEASONS = ["2021", "2020"]  # Add 2021 and 2020 seasons (before 2022)

# Load existing CSV
print("Loading existing CSV...")
df_existing = pd.read_csv('portugal_liga_api_football_20260217.csv')
print(f"Current data: {len(df_existing)} matches\n")

# New matches to add
all_matches = []

for season in SEASONS:
    print(f"Fetching season {season}...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/fixtures",
            headers={"x-apisports-key": API_KEY},
            params={
                "league": LEAGUE_ID,
                "season": season,
                "status": "FT"  # Finished matches only
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['results'] > 0:
                matches = data['response']
                print(f"  Found {len(matches)} finished matches")
                
                for match in matches:
                    # Skip if already in existing data
                    existing_id = df_existing[df_existing['match_id'] == match['fixture']['id']]
                    if not existing_id.empty:
                        continue
                    
                    try:
                        match_data = {
                            'match_id': match['fixture']['id'],
                            'date': match['fixture']['date'],
                            'timestamp': match['fixture']['timestamp'],
                            'season': match['league']['season'],
                            'round': match['league']['round'] if match['league'].get('round') else 'Regular Season',
                            'venue': match['fixture'].get('venue', {}).get('name', 'Unknown'),
                            'city': match['fixture'].get('venue', {}).get('city', 'Unknown'),
                            'referee': match['fixture'].get('referee', 'Unknown'),
                            'status': match['fixture']['status']['long'],
                            'status_short': match['fixture']['status']['short'],
                            'league': match['league']['name'],
                            'country': match['league']['country'],
                            'home_team': match['teams']['home']['name'],
                            'away_team': match['teams']['away']['name'],
                            'home_team_id': match['teams']['home']['id'],
                            'away_team_id': match['teams']['away']['id'],
                            'home_score': match['goals']['home'],
                            'away_score': match['goals']['away'],
                            'home_goals_halftime': match['score']['halftime']['home'],
                            'away_goals_halftime': match['score']['halftime']['away'],
                            'home_goals_fulltime': match['score']['fulltime']['home'],
                            'away_goals_fulltime': match['score']['fulltime']['away'],
                            'match_finished': True
                        }
                        all_matches.append(match_data)
                    except Exception as e:
                        print(f"  Error processing match: {e}")
                        continue
            else:
                print(f"  No finished matches found")
        else:
            print(f"  API Error: {response.status_code}")
            print(f"  Response: {response.json()}")
    
    except Exception as e:
        print(f"  Error fetching season {season}: {e}")
    
    # Rate limiting
    time.sleep(1)

print(f"\nTotal new matches fetched: {len(all_matches)}")

if len(all_matches) > 0:
    # Create dataframe from new matches
    df_new = pd.DataFrame(all_matches)
    
    # Combine with existing
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Remove duplicates by match_id, keeping first occurrence
    df_combined = df_combined.drop_duplicates(subset=['match_id'], keep='first')
    
    # Sort by date
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    # Save
    output_file = f"portugal_liga_api_football_{datetime.now().strftime('%Y%m%d')}_extended.csv"
    df_combined.to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"Original: {len(df_existing)} matches")
    print(f"Added: {len(all_matches)} matches")
    print(f"Total: {len(df_combined)} matches")
    print(f"Saved to: {output_file}")
    
    # Also update the main file
    df_combined.to_csv('portugal_liga_api_football_20260217.csv', index=False)
    print(f"Updated: portugal_liga_api_football_20260217.csv")
else:
    print("No new matches to add")
