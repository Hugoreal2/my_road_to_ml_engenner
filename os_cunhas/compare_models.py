import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Carregar dados
df = pd.read_csv('historico_restaurante.csv')

print("=" * 80)
print("COMPARAÇÃO DE MODELOS - PREVISÃO DE CLIENTES")
print("=" * 80)
print(f"\nDados: {len(df)} dias | Período: {df['data'].min()} a {df['data'].max()}\n")

# Preparar features
df['clientes_media_3dias'] = df['clientes'].rolling(window=3, min_periods=1).mean()

# Converter variáveis categóricas
if 'feriado' in df.columns:
    df['feriado_bin'] = (df['feriado'] == 'Y').astype(int)
if 'ponte' in df.columns:
    df['ponte_bin'] = (df['ponte'] == 'Y').astype(int)
if 'periodo_ordenado' in df.columns:
    df['periodo_ordenado_bin'] = (df['periodo_ordenado'] == 'Y').astype(int)
if 'tem_jogo' in df.columns:
    df['tem_jogo_bin'] = (df['tem_jogo'] == 'Y').astype(int)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['dia_semana'], prefix='dia')

# Features
feature_cols = [col for col in df_encoded.columns if col.startswith('dia_')]
feature_cols.append('clientes_media_3dias')
if 'feriado_bin' in df_encoded.columns:
    feature_cols.extend(['feriado_bin', 'ponte_bin', 'periodo_ordenado_bin', 'tem_jogo_bin'])

X = df_encoded[feature_cols]
y = df_encoded['clientes']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# ============================================
# MODELOS A TESTAR
# ============================================
modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}

# Resultados
resultados = []

print("TREINANDO MODELOS...")
print("-" * 80)

for nome, modelo in modelos.items():
    # Treinar
    modelo.fit(X_train, y_train)
    
    # Prever
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    resultados.append({
        'Modelo': nome,
        'R² Treino': r2_train,
        'R² Teste': r2_test,
        'MAE': mae_test,
        'RMSE': rmse_test,
        'Previsões': y_pred_test
    })
    
    print(f"✓ {nome:20s} | R² Teste: {r2_test:.4f} | MAE: {mae_test:.2f}")

# Criar DataFrame de resultados
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('R² Teste', ascending=False)

print("\n" + "=" * 80)
print("RANKING DOS MODELOS (por R² no Teste)")
print("=" * 80)
print(df_resultados[['Modelo', 'R² Treino', 'R² Teste', 'MAE', 'RMSE']].to_string(index=False))

# Melhor modelo
melhor = df_resultados.iloc[0]
print(f"\n🏆 VENCEDOR: {melhor['Modelo']} com R² = {melhor['R² Teste']:.4f}")

# ============================================
# ANÁLISE DE IMPORTÂNCIA (Random Forest)
# ============================================
print("\n" + "=" * 80)
print("IMPORTÂNCIA DAS FEATURES (Random Forest)")
print("=" * 80)

rf_model = modelos['Random Forest']
importancias = pd.DataFrame({
    'Feature': X.columns,
    'Importância': rf_model.feature_importances_
}).sort_values('Importância', ascending=False)

print(importancias.head(10).to_string(index=False))

# ============================================
# VISUALIZAÇÃO
# ============================================
fig = plt.figure(figsize=(16, 10))

# 1. Comparação R² Score
ax1 = plt.subplot(2, 3, 1)
cores = ['#2ecc71' if r == df_resultados['R² Teste'].max() else '#3498db' 
         for r in df_resultados['R² Teste']]
bars = ax1.barh(df_resultados['Modelo'], df_resultados['R² Teste'], color=cores, alpha=0.8)
ax1.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Comparação de R² (Teste)', fontsize=13, fontweight='bold')
ax1.set_xlim(0, 1)
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_resultados['R² Teste']):
    ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)

# 2. Comparação MAE
ax2 = plt.subplot(2, 3, 2)
cores = ['#2ecc71' if r == df_resultados['MAE'].min() else '#e74c3c' 
         for r in df_resultados['MAE']]
bars = ax2.barh(df_resultados['Modelo'], df_resultados['MAE'], color=cores, alpha=0.8)
ax2.set_xlabel('MAE (clientes)', fontsize=11, fontweight='bold')
ax2.set_title('Erro Médio Absoluto (menor é melhor)', fontsize=13, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, v in enumerate(df_resultados['MAE']):
    ax2.text(v + 0.2, i, f'{v:.1f}', va='center', fontsize=10)

# 3. Overfitting Check (R² Train vs Test)
ax3 = plt.subplot(2, 3, 3)
x_pos = np.arange(len(df_resultados))
width = 0.35
ax3.bar(x_pos - width/2, df_resultados['R² Treino'], width, label='Treino', alpha=0.8, color='#3498db')
ax3.bar(x_pos + width/2, df_resultados['R² Teste'], width, label='Teste', alpha=0.8, color='#e67e22')
ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax3.set_title('Overfitting Check (Treino vs Teste)', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([m[:10] + '...' if len(m) > 10 else m for m in df_resultados['Modelo']], 
                     rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Previsões do Melhor Modelo
ax4 = plt.subplot(2, 3, 4)
melhor_pred = melhor['Previsões']
ax4.plot(range(len(y_test)), y_test.values, 'o-', label='Real', 
         markersize=10, linewidth=2.5, color='#2c3e50')
ax4.plot(range(len(y_test)), melhor_pred, 's-', label='Previsto', 
         markersize=10, linewidth=2.5, alpha=0.7, color='#27ae60')
ax4.set_xlabel('Dia de Teste', fontsize=11, fontweight='bold')
ax4.set_ylabel('Clientes', fontsize=11, fontweight='bold')
ax4.set_title(f'Melhor Modelo: {melhor["Modelo"]}', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Importância Features (Top 8)
ax5 = plt.subplot(2, 3, 5)
top_features = importancias.head(8)
ax5.barh(top_features['Feature'], top_features['Importância'], color='#9b59b6', alpha=0.8)
ax5.set_xlabel('Importância', fontsize=11, fontweight='bold')
ax5.set_title('Top 8 Features Mais Importantes', fontsize=13, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# 6. Distribuição de Erros do Melhor Modelo
ax6 = plt.subplot(2, 3, 6)
erros = y_test.values - melhor_pred
ax6.hist(erros, bins=10, color='#e74c3c', alpha=0.7, edgecolor='black')
ax6.axvline(0, color='black', linestyle='--', linewidth=2)
ax6.set_xlabel('Erro (Real - Previsto)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax6.set_title('Distribuição de Erros', fontsize=13, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
ax6.text(0.02, 0.95, f'Média: {erros.mean():.2f}\nStd: {erros.std():.2f}', 
         transform=ax6.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Gráfico salvo: comparacao_modelos.png")

# ============================================
# CONCLUSÕES
# ============================================
print("\n" + "=" * 80)
print("CONCLUSÕES E RECOMENDAÇÕES")
print("=" * 80)

print(f"\n✓ MELHOR MODELO: {melhor['Modelo']}")
print(f"  - R² Score: {melhor['R² Teste']:.4f} (explica {melhor['R² Teste']*100:.1f}% da variação)")
print(f"  - Erro médio: ±{melhor['MAE']:.1f} clientes")

# Check overfitting
gap = melhor['R² Treino'] - melhor['R² Teste']
if gap > 0.15:
    print(f"\n⚠️  ATENÇÃO: Overfitting detectado! (Gap de {gap:.3f})")
    print("   Sugestão: Mais dados ou regularização")
elif gap < 0.05:
    print(f"\n✓ Modelo bem generalizado (Gap: {gap:.3f})")

print("\n📈 INTERPRETAÇÃO:")
print(f"   Para cada previsão, espera-se um erro de ±{melhor['MAE']:.0f} clientes")
print(f"   Confiança de {melhor['R² Teste']*100:.0f}% nas previsões")

print("\n💡 PRÓXIMOS PASSOS:")
print("   1. Coletar mais dados históricos (3-6 meses)")
print("   2. Adicionar features de clima/temperatura")
print("   3. Testar combinação de modelos (ensemble)")
print("   4. Otimizar hiperparâmetros do melhor modelo")

print("\n" + "=" * 80)
