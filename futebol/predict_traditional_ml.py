"""
Soccer Match Predictor - Feature Engineering and Model Training
This script will prepare features and train a prediction model once you have sufficient data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_data(filename):
    """Load match data from CSV"""
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    return df

def engineer_features(df):
    """Create features for prediction"""
    
    # Convert scores to numeric
    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
    
    # Filter only finished matches with valid scores
    df = df[(df['match_finished'] == True) & 
            (df['home_score'].notna()) & 
            (df['away_score'].notna())].copy()
    
    if len(df) == 0:
        print("No finished matches with valid scores found!")
        return None
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create target variable (match result)
    # 0 = Away Win, 1 = Draw, 2 = Home Win
    df['result'] = np.where(df['home_score'] > df['away_score'], 2,
                   np.where(df['home_score'] < df['away_score'], 0, 1))
    
    # Calculate goal difference
    df['goal_difference'] = df['home_score'] - df['away_score']
    df['total_goals'] = df['home_score'] + df['away_score']
    
    # Feature 1: Home advantage (historical)
    home_teams = df.groupby('home_team').agg({
        'result': lambda x: (x == 2).mean(),  # Home win rate
        'home_score': 'mean',
        'away_score': 'mean'
    }).rename(columns={
        'result': 'home_win_rate',
        'home_score': 'home_avg_scored',
        'away_score': 'home_avg_conceded'
    })
    
    # Feature 2: Away performance
    away_teams = df.groupby('away_team').agg({
        'result': lambda x: (x == 0).mean(),  # Away win rate
        'away_score': 'mean',
        'home_score': 'mean'
    }).rename(columns={
        'result': 'away_win_rate',
        'away_score': 'away_avg_scored',
        'home_score': 'away_avg_conceded'
    })
    
    # Merge features
    df = df.merge(home_teams, left_on='home_team', right_index=True, how='left')
    df = df.merge(away_teams, left_on='away_team', right_index=True, how='left')
    
    # Fill NaN values with averages
    df = df.fillna(df.mean(numeric_only=True))
    
    # Feature 3: Recent form (last 5 matches)
    def calculate_recent_form(team_df, n=5):
        """Calculate form based on last n matches"""
        if len(team_df) < n:
            return team_df['result'].map({2: 3, 1: 1, 0: 0}).sum() / len(team_df)
        return team_df.tail(n)['result'].map({2: 3, 1: 1, 0: 0}).sum() / n
    
    # Calculate form for home teams
    home_form = []
    for idx, row in df.iterrows():
        team_matches = df[(df['home_team'] == row['home_team']) | 
                         (df['away_team'] == row['home_team'])]
        team_matches = team_matches[team_matches['date'] < row['date']]
        
        if len(team_matches) == 0:
            home_form.append(1.5)  # Neutral form
        else:
            recent = team_matches.tail(5)
            points = 0
            for _, match in recent.iterrows():
                if match['home_team'] == row['home_team']:
                    points += {2: 3, 1: 1, 0: 0}[match['result']]
                else:
                    points += {0: 3, 1: 1, 2: 0}[match['result']]
            home_form.append(points / min(len(recent), 5))
    
    df['home_recent_form'] = home_form
    
    # Calculate form for away teams
    away_form = []
    for idx, row in df.iterrows():
        team_matches = df[(df['home_team'] == row['away_team']) | 
                         (df['away_team'] == row['away_team'])]
        team_matches = team_matches[team_matches['date'] < row['date']]
        
        if len(team_matches) == 0:
            away_form.append(1.5)  # Neutral form
        else:
            recent = team_matches.tail(5)
            points = 0
            for _, match in recent.iterrows():
                if match['home_team'] == row['away_team']:
                    points += {2: 3, 1: 1, 0: 0}[match['result']]
                else:
                    points += {0: 3, 1: 1, 2: 0}[match['result']]
            away_form.append(points / min(len(recent), 5))
    
    df['away_recent_form'] = away_form
    
    return df

def prepare_training_data(df):
    """Prepare features and target for training"""
    
    # Select features for training
    feature_columns = [
        'home_win_rate', 'home_avg_scored', 'home_avg_conceded',
        'away_win_rate', 'away_avg_scored', 'away_avg_conceded',
        'home_recent_form', 'away_recent_form'
    ]
    
    X = df[feature_columns]
    y = df['result']
    
    return X, y, feature_columns

def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple models and compare performance"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 70)
        
        # Train
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Away Win', 'Draw', 'Home Win'],
                                   zero_division=0))
        
        # Feature importance (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Feature Importance:")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']:<25} {row['importance']:.4f}")
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    return results, scaler

def main():
    """Main function"""
    
    print("=" * 70)
    print("SOCCER MATCH PREDICTION - MACHINE LEARNING MODEL")
    print("=" * 70)
    
    # Load data
    print("\nStep 1: Loading data...")
    try:
        df = load_data('portugal_liga_api_football_20260217.csv')
        print(f"✓ Loaded {len(df)} matches")
    except FileNotFoundError:
        print("✗ CSV file not found. Please run fetch_api_football.py first!")
        return
    
    # Engineer features
    print("\nStep 2: Engineering features...")
    df = engineer_features(df)
    
    if df is None or len(df) < 20:
        print("✗ Insufficient data for training!")
        print("  Note: The free API key has limitations.")
        print("  Consider upgrading to premium or collecting more data over time.")
        return
    
    print(f"✓ Prepared {len(df)} matches with features")
    print(f"  - Result distribution:")
    print(f"    Home Wins: {(df['result'] == 2).sum()}")
    print(f"    Draws: {(df['result'] == 1).sum()}")
    print(f"    Away Wins: {(df['result'] == 0).sum()}")
    
    # Prepare training data
    print("\nStep 3: Preparing training data...")
    X, y, feature_names = prepare_training_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {len(X_train)} matches")
    print(f"✓ Test set: {len(X_test)} matches")
    
    # Train models
    print("\nStep 4: Training models...")
    results, scaler = train_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("=" * 70)
    
    print("\nNote: This is a basic model. For better predictions, consider:")
    print("  - More historical data (multiple seasons)")
    print("  - Head-to-head records")
    print("  - Player statistics and injuries")
    print("  - Weather conditions")
    print("  - Betting odds")
    print("  - Home/away streaks")

if __name__ == "__main__":
    main()
