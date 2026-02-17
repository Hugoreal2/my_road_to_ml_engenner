"""
Generate synthetic historical data for Portuguese Liga (2010-2022)
Creates realistic matches based on existing team patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Load existing data to learn patterns
print("Loading existing data for pattern analysis...")
df_existing = pd.read_csv('portugal_liga_api_football_20260217.csv')
df_existing['date'] = pd.to_datetime(df_existing['date'])

# Extract team names
teams = sorted(set(df_existing['home_team'].unique().tolist() + df_existing['away_team'].unique().tolist()))
print(f"Found {len(teams)} unique teams")
print(f"Teams: {teams}\n")

# Calculate team strength from existing data
team_stats = {}
for team in teams:
    home_matches = df_existing[df_existing['home_team'] == team]
    away_matches = df_existing[df_existing['away_team'] == team]
    
    home_goals_for = home_matches['home_score'].mean() if len(home_matches) > 0 else 1.4
    home_goals_against = home_matches['away_score'].mean() if len(home_matches) > 0 else 1.1
    away_goals_for = away_matches['away_score'].mean() if len(away_matches) > 0 else 1.0
    away_goals_against = away_matches['home_score'].mean() if len(away_matches) > 0 else 1.3
    
    home_wins = len(home_matches[home_matches['home_score'] > home_matches['away_score']])
    away_wins = len(away_matches[away_matches['away_score'] > away_matches['home_score']])
    total_wins = home_wins + away_wins
    total_games = len(home_matches) + len(away_matches)
    
    team_stats[team] = {
        'home_goals_for': home_goals_for,
        'home_goals_against': home_goals_against,
        'away_goals_for': away_goals_for,
        'away_goals_against': away_goals_against,
        'win_rate': total_wins / total_games if total_games > 0 else 0.4,
        'strength': (total_wins / total_games) if total_games > 0 else 0.5
    }

print("Team strength analysis:")
for team, stats in sorted(team_stats.items(), key=lambda x: x[1]['strength'], reverse=True)[:5]:
    print(f"  {team:20s}: strength={stats['strength']:.2f}")
print()

# Generate synthetic data from 2010 to 2021
print("Generating synthetic matches from 2010-2021...")

synthetic_matches = []
match_id_start = 900000

# Portuguese Liga typically has 34 rounds per season, 10 teams (varies)
# Each round has ~5 matches (with 18 teams now, it's 9 matches per round)
# Historical: 2010-2019 had 16-18 teams, 2020+ has 18 teams

seasons_data = {
    2010: (16, 30),  # (num_teams, rounds)
    2011: (16, 30),
    2012: (16, 30),
    2013: (16, 30),
    2014: (16, 30),
    2015: (16, 30),
    2016: (18, 34),
    2017: (18, 34),
    2018: (18, 34),
    2019: (18, 34),
    2020: (18, 34),
    2021: (18, 34),
}

match_id = match_id_start
venues_data = {
    'Benfica': ('Estádio do Sport Lisboa e Benfica', 'Lisboa'),
    'FC Porto': ('Estádio do Dragão', 'Porto'),
    'Sporting CP': ('Estádio José Alvalade', 'Lisboa'),
    'SC Braga': ('Estádio Municipal de Braga', 'Braga'),
    'Vitória SC': ('Estádio D. Afonso Henriques', 'Guimarães'),
    'Rio Ave': ('Estádio do Rio Ave Futebol Clube', 'Vila do Conde'),
    'Boavista': ('Estádio do Bessa Século XXI', 'Porto'),
    'Estoril': ('Estádio António Coimbra da Mota', 'Estoril'),
    'Marítimo': ('Estádio dos Barreiros', 'Madeira'),
    'GIL Vicente': ('Estádio Cidade de Barcelos', 'Barcelos'),
    'Casa Pia': ('Estádio Nacional', 'Jamor'),
    'Portimonense': ('Estádio Municipal de Portimão', 'Portimão'),
    'Santa Clara': ('Estádio de São Miguel', 'Açores'),
    'Paços Ferreira': ('Estádio da Capital do Móvel', 'Paços Ferreira'),
    'Famalicão': ('Estádio Municipal 22 de Junho', 'Famalicão'),
    'Arouca': ('Estádio Municipal de Arouca', 'Arouca'),
    'Vizela': ('Estádio do Vizela', 'Vizela'),
    'Chaves': ('Estádio Municipal Eng. Manuel Branco Teixeira', 'Chaves'),
}

referees = [
    'Manuel Mota', 'Fábio Veríssimo', 'Gustavo Correia', 'Cláudio Pereira',
    'Tiago Martins', 'João Pedro Pinheiro', 'Hélder Malheiro', 'André Narciso',
    'Carlos Macedo', 'Nuno Miguel Serrano Almeida', 'Rui Costa', 'Luis Godinho'
]

for season in sorted(seasons_data.keys()):
    num_teams, num_rounds = seasons_data[season]
    
    # Get teams available in that season (use a subset of current teams)
    available_teams = teams[:num_teams]
    
    print(f"  Season {season}: {num_teams} teams, {num_rounds} rounds")
    
    # Generate matches for each round
    for round_num in range(1, num_rounds + 1):
        # Create all possible matchups for the round
        shuffle_teams = available_teams.copy()
        random.shuffle(shuffle_teams)
        
        # Pair teams for matches
        num_matches = len(shuffle_teams) // 2
        
        for match_idx in range(num_matches):
            home_team = shuffle_teams[match_idx * 2]
            away_team = shuffle_teams[match_idx * 2 + 1]
            
            # Generate realistic score
            home_strength = team_stats[home_team]['strength']
            away_strength = team_stats[away_team]['strength']
            
            # Home advantage: add 0.3 to home strength
            home_advantage = 0.3
            adjusted_home_strength = home_strength + home_advantage
            total_strength = adjusted_home_strength + away_strength
            
            # Normalize to 0-1
            home_win_prob = adjusted_home_strength / (total_strength + 0.1)
            
            # Generate goals based on strength
            home_goals = np.random.poisson(1.4 * home_strength)
            away_goals = np.random.poisson(1.0 * away_strength)
            
            # Halftime goals (usually lower)
            home_ht = max(0, home_goals - np.random.randint(0, 2))
            away_ht = max(0, away_goals - np.random.randint(0, 2))
            
            # Generate date (spread throughout season)
            season_start = datetime(season, 8, 1)
            days_into_season = (round_num - 1) * 7 + np.random.randint(0, 7)
            match_date = season_start + timedelta(days=days_into_season)
            
            # Get venue from home team
            if home_team in venues_data:
                venue, city = venues_data[home_team]
            else:
                venue = f"Estádio de {home_team}"
                city = "Portugal"
            
            synthetic_match = {
                'match_id': match_id,
                'date': match_date.strftime('%Y-%m-%d %H:%M:%S') + '+00:00',
                'timestamp': int(match_date.timestamp()),
                'season': season,
                'round': f'Regular Season - {round_num}',
                'venue': venue,
                'city': city,
                'referee': random.choice(referees),
                'status': 'Match Finished',
                'status_short': 'FT',
                'league': 'Primeira Liga',
                'country': 'Portugal',
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': hash(home_team) % 10000,
                'away_team_id': hash(away_team) % 10000,
                'home_score': home_goals,
                'away_score': away_goals,
                'home_goals_halftime': home_ht,
                'away_goals_halftime': away_ht,
                'home_goals_fulltime': home_goals,
                'away_goals_fulltime': away_goals,
                'match_finished': True
            }
            
            synthetic_matches.append(synthetic_match)
            match_id += 1

print(f"\nGenerated {len(synthetic_matches)} synthetic matches\n")

# Create dataframe
df_synthetic = pd.DataFrame(synthetic_matches)

# Load original data and combine
df_original = pd.read_csv('portugal_liga_api_football_20260217.csv')
df_combined = pd.concat([df_synthetic, df_original], ignore_index=True)

# Sort by date
df_combined['date'] = pd.to_datetime(df_combined['date'])
df_combined = df_combined.sort_values('date').reset_index(drop=True)

# Remove any duplicates by match_id
df_combined = df_combined.drop_duplicates(subset=['match_id'], keep='first')

# Save
output_file = 'portugal_liga_api_football_20260217_extended.csv'
df_combined.to_csv(output_file, index=False)

print("="*60)
print("✅ DATASET EXPANDED!")
print("="*60)
print(f"Original matches: {len(df_original)}")
print(f"Synthetic matches: {len(df_synthetic)}")
print(f"Total matches: {len(df_combined)}")
print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")
print(f"\nSaved to: {output_file}")
print("\nNow copy this file to replace the original:")
print(f"  cp {output_file} portugal_liga_api_football_20260217.csv")
