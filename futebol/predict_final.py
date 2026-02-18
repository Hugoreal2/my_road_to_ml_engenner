"""
Soccer Match Predictor - FINAL VERSION
Using Logistic Regression (best model found)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("SOCCER MATCH PREDICTOR - FINAL (LOGISTIC REGRESSION)")
print("="*70 + "\n")

# ============================================
# LOAD & SPLIT DATA
# ============================================

print("Loading data...")
df = pd.read_csv('portugal_liga_api_football_20260217.csv')
df = df[df['match_finished'] == True].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"✓ {len(df)} finished matches")

# Target
df['result'] = np.where(df['home_score'] > df['away_score'], 2,
                        np.where(df['home_score'] < df['away_score'], 0, 1))

print(f"  Home Win: {(df['result'] == 2).sum()} | Draw: {(df['result'] == 1).sum()} | Away Win: {(df['result'] == 0).sum()}\n")

# Split FIRST
split = int(len(df) * 0.8)
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

print(f"Train: {len(df_train)} | Test: {len(df_test)}\n")

# ============================================
# FEATURE ENGINEERING (training data only)
# ============================================

print("Computing features (from training data only)...")

def get_team_form(df_hist, team, date, n=5):
    """Get form score based on last n matches"""
    prev = df_hist[((df_hist['home_team'] == team) | (df_hist['away_team'] == team)) & 
                    (df_hist['date'] < date)].tail(n)
    
    if len(prev) == 0:
        return 1.5
    
    points = 0
    for _, m in prev.iterrows():
        if m['home_team'] == team:
            points += 3 if m['home_score'] > m['away_score'] else 1 if m['home_score'] == m['away_score'] else 0
        else:
            points += 3 if m['away_score'] > m['home_score'] else 1 if m['away_score'] == m['home_score'] else 0
    
    return points / min(len(prev), n)

# Calculate from training data only
home_goals_train = df_train.groupby('home_team')['home_score'].mean().to_dict()
away_goals_train = df_train.groupby('away_team')['away_score'].mean().to_dict()
home_conceded_train = df_train.groupby('home_team')['away_score'].mean().to_dict()
away_conceded_train = df_train.groupby('away_team')['home_score'].mean().to_dict()

# Calculate wins/draws/losses rates
home_win_rate_train = (df_train.groupby('home_team')['result'].apply(lambda x: (x == 2).sum()) / 
                       df_train.groupby('home_team').size()).to_dict()
away_win_rate_train = (df_train.groupby('away_team')['result'].apply(lambda x: (x == 0).sum()) / 
                       df_train.groupby('away_team').size()).to_dict()

# Add to train
train_home_form = []
train_away_form = []
for idx, row in df_train.iterrows():
    train_home_form.append(get_team_form(df_train, row['home_team'], row['date'], 5))
    train_away_form.append(get_team_form(df_train, row['away_team'], row['date'], 5))

df_train['home_form'] = train_home_form
df_train['away_form'] = train_away_form
df_train['home_avg_scored'] = df_train['home_team'].map(home_goals_train).fillna(1.5)
df_train['away_avg_scored'] = df_train['away_team'].map(away_goals_train).fillna(1.5)
df_train['home_avg_conceded'] = df_train['home_team'].map(home_conceded_train).fillna(1.0)
df_train['away_avg_conceded'] = df_train['away_team'].map(away_conceded_train).fillna(1.0)
df_train['home_win_rate'] = df_train['home_team'].map(home_win_rate_train).fillna(0.5)
df_train['away_win_rate'] = df_train['away_team'].map(away_win_rate_train).fillna(0.5)

# Add to test (using training stats)
test_home_form = []
test_away_form = []
for idx, row in df_test.iterrows():
    test_home_form.append(get_team_form(df_train, row['home_team'], row['date'], 5))
    test_away_form.append(get_team_form(df_train, row['away_team'], row['date'], 5))

df_test['home_form'] = test_home_form
df_test['away_form'] = test_away_form
df_test['home_avg_scored'] = df_test['home_team'].map(home_goals_train).fillna(1.5)
df_test['away_avg_scored'] = df_test['away_team'].map(away_goals_train).fillna(1.5)
df_test['home_avg_conceded'] = df_test['home_team'].map(home_conceded_train).fillna(1.0)
df_test['away_avg_conceded'] = df_test['away_team'].map(away_conceded_train).fillna(1.0)
df_test['home_win_rate'] = df_test['home_team'].map(home_win_rate_train).fillna(0.5)
df_test['away_win_rate'] = df_test['away_team'].map(away_win_rate_train).fillna(0.5)

print("✓ Features computed\n")

# ============================================
# PREPARE DATA
# ============================================

features = ['home_form', 'away_form', 'home_avg_scored', 'away_avg_scored',
            'home_avg_conceded', 'away_avg_conceded', 'home_win_rate', 'away_win_rate']

X_train = df_train[features].values
y_train = df_train['result'].values

X_test = df_test[features].values
y_test = df_test['result'].values

# Normalize (fit on training data only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Features: {len(features)}")
for i, f in enumerate(features, 1):
    print(f"  {i}. {f}")
print()

# ============================================
# TRAIN LOGISTIC REGRESSION
# ============================================

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("✓ Model trained\n")

# ============================================
# EVALUATE
# ============================================

print("="*70)
print("RESULTS")
print("="*70 + "\n")

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 ACCURACY: {accuracy:.2%}\n")

labels = ['Away Win', 'Draw', 'Home Win']
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                Away  Draw  Home")
for i, name in enumerate(labels):
    print(f"{name:12s} [{cm[i][0]:4d}  {cm[i][1]:4d}  {cm[i][2]:4d}]")

print()
for i, name in enumerate(labels):
    total = cm[i].sum()
    correct = cm[i][i]
    pct = correct / total * 100 if total > 0 else 0
    print(f"{name:12s}: {correct:3d}/{total:3d} ({pct:5.1f}%)")

print("\n" + "="*70)
baseline = 0.5714  # Always predict home win
improvement = (accuracy - baseline) * 100

print(f"Baseline (always Home Win): 57.14%")
print(f"Logistic Regression:        {accuracy:.2%}")
print(f"Delta:                      {improvement:+.2f}pp")

if accuracy > baseline:
    print("✅ BETTER than baseline!")
else:
    print("❌ Worse than baseline")

print("="*70)
print(f"\nModel coefficients (feature importance):")
for feat, coef in sorted(zip(features, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat:20s}: {coef:+.4f}")

print("\n✓ Done!\n")
