import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('historico_restaurante.csv')

print("=" * 60)
print("MODELO DE PREVISÃO DE CLIENTES - RESTAURANTE")
print("=" * 60)
print(f"\nDados carregados: {len(df)} dias de histórico")
print(f"Período: {df['data'].min()} a {df['data'].max()}\n")

# Preparar features
# Criar média móvel de clientes dos últimos 3 dias ANTES do encoding
df['clientes_media_3dias'] = df['clientes'].rolling(window=3, min_periods=1).mean()

# Converter variáveis categóricas para numéricas
# Feriado, ponte, periodo_ordenado, tem_jogo -> 0 ou 1
if 'feriado' in df.columns:
    df['feriado_bin'] = (df['feriado'] == 'Y').astype(int)
if 'ponte' in df.columns:
    df['ponte_bin'] = (df['ponte'] == 'Y').astype(int)
if 'periodo_ordenado' in df.columns:
    df['periodo_ordenado_bin'] = (df['periodo_ordenado'] == 'Y').astype(int)
if 'tem_jogo' in df.columns:
    df['tem_jogo_bin'] = (df['tem_jogo'] == 'Y').astype(int)

# Converter dia da semana para variável numérica (One-Hot Encoding)
df_encoded = pd.get_dummies(df, columns=['dia_semana'], prefix='dia')

# Preparar dataset
feature_cols = [col for col in df_encoded.columns if col.startswith('dia_')]
feature_cols.append('clientes_media_3dias')

# Adicionar novas features se existirem
if 'feriado_bin' in df_encoded.columns:
    feature_cols.extend(['feriado_bin', 'ponte_bin', 'periodo_ordenado_bin', 'tem_jogo_bin'])

X = df_encoded[feature_cols]
y = df_encoded['clientes']

# Split treino/teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

print("TREINO DO MODELO")
print("-" * 60)
print(f"Dados de treino: {len(X_train)} dias")
print(f"Dados de teste: {len(X_test)} dias")

# Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Avaliar modelo
print("\nRESULTADOS DO MODELO")
print("-" * 60)
print("TREINO:")
print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"  MAE (Erro Médio Absoluto): {mean_absolute_error(y_train, y_pred_train):.2f} clientes")
print(f"  RMSE (Raiz do Erro Quadrático): {np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f} clientes")

print("\nTESTE:")
print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"  MAE (Erro Médio Absoluto): {mean_absolute_error(y_test, y_pred_test):.2f} clientes")
print(f"  RMSE (Raiz do Erro Quadrático): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} clientes")

# Importância das features
print("\nIMPORTÂNCIA DAS VARIÁVEIS")
print("-" * 60)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coeficiente': model.coef_
}).sort_values('Coeficiente', ascending=False)
print(feature_importance.to_string(index=False))

# Previsão para próximos dias
print("\n" + "=" * 60)
print("PREVISÃO PARA PRÓXIMA SEMANA")
print("=" * 60)

dias_semana = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ultima_media = df['clientes'].tail(3).mean()

for dia in dias_semana:
    # Criar features para o dia
    nova_observacao = pd.DataFrame(0, index=[0], columns=X.columns)
    nova_observacao[f'dia_{dia}'] = 1
    nova_observacao['clientes_media_3dias'] = ultima_media
    
    previsao = model.predict(nova_observacao)[0]
    print(f"{dia:12s}: {previsao:.0f} clientes estimados")

# Visualização
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Gráfico 1: Real vs Previsto (Teste)
axes[0].plot(range(len(y_test)), y_test.values, 'o-', label='Real', markersize=8, linewidth=2)
axes[0].plot(range(len(y_test)), y_pred_test, 's-', label='Previsto', markersize=8, linewidth=2, alpha=0.7)
axes[0].set_xlabel('Dia de Teste', fontsize=12)
axes[0].set_ylabel('Número de Clientes', fontsize=12)
axes[0].set_title('Previsão vs Real (Dados de Teste)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Distribuição por dia da semana
df_semana = df.groupby('dia_semana')['clientes'].mean().reindex(dias_semana)
axes[1].bar(range(len(dias_semana)), df_semana.values, color='steelblue', alpha=0.7)
axes[1].set_xticks(range(len(dias_semana)))
axes[1].set_xticklabels(dias_semana, rotation=45, ha='right')
axes[1].set_xlabel('Dia da Semana', fontsize=12)
axes[1].set_ylabel('Média de Clientes', fontsize=12)
axes[1].set_title('Média de Clientes por Dia da Semana', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('previsao_clientes.png', dpi=300, bbox_inches='tight')
print(f"\n📊 Gráfico salvo: previsao_clientes.png")

print("\n" + "=" * 60)
print("INTERPRETAÇÃO DOS RESULTADOS")
print("=" * 60)
print("""
✓ R² Score: Quanto mais próximo de 1, melhor o modelo explica os dados
✓ MAE: Erro médio em número de clientes (quanto menor, melhor)
✓ RMSE: Penaliza erros maiores (quanto menor, melhor)

""")
