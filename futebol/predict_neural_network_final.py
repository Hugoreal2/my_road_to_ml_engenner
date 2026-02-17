"""
Soccer Match Predictor - Neural Network (FIXED)
Simplified and fixed to actually work
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("SOCCER MATCH PREDICTOR - NEURAL NETWORK (FIXED)")
print("="*70 + "\n")

# ============================================
# LOAD DATA
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

# ============================================
# SIMPLE FEATURES - FOCUS ON WHAT MATTERS
# ============================================

print("Computing features...")

# Home advantage
home_stats = df.groupby('home_team').agg({
    'result': lambda x: ((x == 2).sum(), (x == 1).sum(), (x == 0).sum())
}).to_dict()

away_stats = df.groupby('away_team').agg({
    'result': lambda x: ((x == 2).sum(), (x == 1).sum(), (x == 0).sum())
}).to_dict()

# Simple form: last N matches
def get_team_form(df, team, date, n=3):
    """Get simple form score (0-3) based on last n matches"""
    prev = df[((df['home_team'] == team) | (df['away_team'] == team)) & (df['date'] < date)].tail(n)
    
    if len(prev) == 0:
        return 1.5  # Neutral
    
    points = 0
    for _, m in prev.iterrows():
        if m['home_team'] == team:
            points += 3 if m['home_score'] > m['away_score'] else 1 if m['home_score'] == m['away_score'] else 0
        else:
            points += 3 if m['away_score'] > m['home_score'] else 1 if m['away_score'] == m['home_score'] else 0
    
    return points / min(len(prev), n)

# Calculate form for each match
home_form = []
away_form = []

for idx, row in df.iterrows():
    home_form.append(get_team_form(df, row['home_team'], row['date'], 5))
    away_form.append(get_team_form(df, row['away_team'], row['date'], 5))

df['home_form'] = home_form
df['away_form'] = away_form

# Simple goal averages
home_goals = df.groupby('home_team')['home_score'].mean().to_dict()
away_goals = df.groupby('away_team')['away_score'].mean().to_dict()
home_conceded = df.groupby('home_team')['away_score'].mean().to_dict()
away_conceded = df.groupby('away_team')['home_score'].mean().to_dict()

df['home_avg_goals'] = df['home_team'].map(home_goals).fillna(1.5)
df['away_avg_goals'] = df['away_team'].map(away_goals).fillna(1.5)
df['home_avg_conceded'] = df['home_team'].map(home_conceded).fillna(1.0)
df['away_avg_conceded'] = df['away_team'].map(away_conceded).fillna(1.0)

# Goal difference in form
df['goal_diff'] = (df['home_avg_goals'] - df['home_avg_conceded']) - (df['away_avg_goals'] - df['away_avg_conceded'])

print("✓ Features computed\n")

# ============================================
# PREPARE DATA
# ============================================

features = ['home_form', 'away_form', 'home_avg_goals', 'away_avg_goals', 
            'home_avg_conceded', 'away_avg_conceded', 'goal_diff']

X = df[features].values
y = df['result'].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time split (80/20)
split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# ============================================
# BUILD SIMPLER MODEL
# ============================================

print("Building model...")

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(len(features),)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(16, activation='relu'),
    
    layers.Dense(3, activation='softmax')
])

# Use categorical crossentropy with one-hot encoding
y_train_cat = keras.utils.to_categorical(y_train, 3)
y_test_cat = keras.utils.to_categorical(y_test, 3)

model.compile(
    optimizer=keras.optimizers.Adam(0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training...\n")

history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
    verbose=0
)

# ============================================
# EVALUATE
# ============================================

print("="*70)
print("RESULTS")
print("="*70 + "\n")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"🎯 ACCURACY: {acc:.2%}\n")

labels = ['Away Win', 'Draw', 'Home Win']
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
improvement = (acc - 0.5714) * 100
print(f"Baseline: 57.14%")
print(f"Model:    {acc:.2%}")
print(f"Delta:    {improvement:+.2f}pp")
if acc > 0.5714:
    print("✅ BETTER than baseline!")
elif acc < 0.5714:
    print("❌ Worse than baseline")
else:
    print("➖ Same as baseline")
print("="*70 + "\n")
