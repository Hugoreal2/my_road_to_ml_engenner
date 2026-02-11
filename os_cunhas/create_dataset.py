import pandas as pd
import random
from datetime import datetime, timedelta

print("=" * 80)
print("GERAÇÃO DE DATASET COM FEATURES")
print("=" * 80)

# ============================================
# PARTE 1: GERAR DATASET
# ============================================
print("\n[1/2] Gerando dataset sintético...")

menu_items = [
    "SOPA DO DIA","CALDINHO CAMARÃO","PÃO","MANTEIGAS/PATÊS","PÃO TORRADO",
    "AMEIJOA (DOSE)","CAMARÃO FRITO","PRATO CAMARÃO","PICA-PAU",
    "BITOQUE VACA","BITOQUE PORCO","BIFE NOVILHO","BIFE À INGLESA",
    "PREGO NO PÃO","PREGO NO PÃO C/ QUEIJO","PREGO NO PÃO C/ FIAMBRE","PREGO NO PÃO C/ OVO",
    "PREGO NO PÃO C/ BACON","PREGO NO PÃO MISTO C/ OVO","PREGO NO PÃO ESPECIAL","PREGO NO PÃO C/ CEBOLA",
    "OMELETE SIMPLES","OMELETE CAMARÃO","OMELETE FIAMBRE","OMELETE QUEIJO","OMELETE MISTA",
    "BATATA (DOSE)","ARROZ (DOSE)","SALADA (ALFACE OU TOMATE)","SALADA (MISTA)",
    "MOUSSE CHOCOLATE","DOCE DA CASA","BABA DE CAMELO","PUDIM","LEITE CREME","SALADA DE FRUTAS",
    "VINHO VERDE (JARRO ½ L)","VINHO VERDE (JARRO 1 L)","VINHO VERDE (COPO)",
    "VINHO TINTO (JARRO ½ L)","VINHO TINTO (JARRO 1 L)","VINHO TINTO (COPO)",
    "EA (PEQUENA)","EA (GRANDE)","BORBA (PEQUENA)","BORBA (GRANDE)","MARQUÊS DE BORBA",
    "MONTE DE PINHEIROS (BAT)","ÁGUA","ÁGUA C/ SABORES","COCA-COLA","SUMOL ANANÁS",
    "SUMOL LARANJA","NECTAR","BI","GUARANÁ","7 UP","SAGRES (MÉDIA)","IMPERIAL",
    "SUPER BOCK (MÉDIA)","SUPER BOCK STOUT (MÉDIA)","CERVEJA S/ ALCOOL",
    "GRANTS ½","GRANTS 1","FAMOUS ½","FAMOUS 1","JAMESON ½","JAMESON 1","JB ½","JB 1",
    "ROCHEDO","MACIEIRA ½","MACIEIRA 1","MONTES DE SAUDADE ½","MONTES DE SAUDADE 1",
    "LICOR BEIRÃO ½","LICOR BEIRÃO 1","AMÊNDOA AMARGA"
]

def gerar_vendas(clientes):
    vendas = {}
    for item in menu_items:
        if "PÃO" in item or "PREGO" in item or "OMELETE" in item or "BITOQUE" in item:
            vendas[item] = random.randint(max(1,int(clientes*0.1)), max(2,int(clientes*0.3)))
        elif "CAMARÃO" in item or "BIFE" in item or "AMEIJOA" in item:
            vendas[item] = random.randint(max(0,int(clientes*0.05)), max(1,int(clientes*0.2)))
        elif "VINHO" in item or "EA" in item or "BORBA" in item:
            vendas[item] = random.randint(max(0,int(clientes*0.02)), max(1,int(clientes*0.1)))
        elif "DOCES" in item or "SALADA" in item:
            vendas[item] = random.randint(max(0,int(clientes*0.05)), max(1,int(clientes*0.15)))
        else:
            vendas[item] = random.randint(max(0,int(clientes*0.05)), max(1,int(clientes*0.2)))
    return vendas

start_date = datetime(2025, 8, 1)  # 6 meses atrás
historico = []

# Lista de jogos importantes (6 meses)
jogos_futebol_calendario = {
    # Agosto 2025
    '2025-08-03': 'Benfica vs Porto',
    '2025-08-10': 'Sporting vs Braga',
    '2025-08-24': 'Porto vs Sporting',
    '2025-08-31': 'Benfica vs Sporting',
    # Setembro 2025
    '2025-09-07': 'Porto vs Benfica',
    '2025-09-14': 'Sporting vs Benfica',
    '2025-09-21': 'Benfica vs Braga',
    '2025-09-28': 'Porto vs Vitória',
    # Outubro 2025
    '2025-10-05': 'Benfica vs Sporting',
    '2025-10-12': 'Porto vs Braga',
    '2025-10-19': 'Sporting vs Porto',
    '2025-10-26': 'Benfica vs Vitória',
    # Novembro 2025
    '2025-11-02': 'Porto vs Sporting',
    '2025-11-09': 'Benfica vs Braga',
    '2025-11-16': 'Sporting vs Benfica',
    '2025-11-23': 'Porto vs Benfica',
    '2025-11-30': 'Sporting vs Braga',
    # Dezembro 2025
    '2025-12-07': 'Benfica vs Porto',
    '2025-12-14': 'Porto vs Sporting',
    '2025-12-21': 'Sporting vs Benfica',
    '2025-12-28': 'Benfica vs Braga',
    # Janeiro 2026
    '2026-01-04': 'Benfica vs Porto',
    '2026-01-11': 'Sporting vs Braga',
    '2026-01-18': 'Porto vs Sporting',
    '2026-01-25': 'Benfica vs Sporting',
}

# Padrões de chuva (mais chuva no inverno)
def tem_chuva(mes):
    """Portugal: Mais chuva Out-Mar, menos Jun-Set"""
    if mes in [6, 7, 8, 9]:  # Verão
        return random.random() < 0.15  # 15% chance
    elif mes in [11, 12, 1, 2, 3]:  # Inverno
        return random.random() < 0.50  # 50% chance
    else:  # Primavera/Outono
        return random.random() < 0.35  # 35% chance

for i in range(183):  # ~6 meses (183 dias)
    data = start_date + timedelta(days=i)
    dia_semana = data.strftime("%A")
    data_str = data.strftime("%Y-%m-%d")
    mes = data.month
    
    # Base de clientes
    base_clientes = 0
    if dia_semana in ["Saturday", "Sunday"]:
        base_clientes = random.randint(80, 130)
    else:
        base_clientes = random.randint(30, 60)
    
    # Ajustes baseados em contexto
    choveu = tem_chuva(mes)
    tem_jogo = data_str in jogos_futebol_calendario
    
    # Chuva aumenta clientes (+ 10-20%)
    if choveu:
        base_clientes = int(base_clientes * random.uniform(1.1, 1.2))
    
    # Jogos aumentam clientes (+ 15-30%)
    if tem_jogo:
        base_clientes = int(base_clientes * random.uniform(1.15, 1.3))
    
    # Verão (Jun-Set): +10% movimento
    if mes in [6, 7, 8, 9]:
        base_clientes = int(base_clientes * 1.1)
    
    # Natal/Ano Novo: +20-40%
    if mes == 12 and data.day >= 20:
        base_clientes = int(base_clientes * random.uniform(1.2, 1.4))
    
    clientes = min(base_clientes, 150)  # Cap máximo 150 clientes
    
    vendas = gerar_vendas(clientes)
    historico.append({
        "data": data_str,
        "dia_semana": dia_semana,
        "clientes": clientes,
        "choveu": 'Y' if choveu else 'N',
        "temperatura": random.randint(8, 35) if mes in [6,7,8] else random.randint(5, 25),
        **vendas
    })

df = pd.DataFrame(historico)
df.to_csv("historico_restaurante.csv", index=False)
print(f"   ✓ CSV gerado com {len(df)} dias de dados")

# ============================================
# PARTE 2: ADICIONAR FEATURES
# ============================================
print("\n[2/2] Adicionando features contextuais...")

df['data'] = pd.to_datetime(df['data'])

# Feriados Portugal 2025-2026
feriados_2026 = pd.to_datetime([
    # 2025
    '2025-08-15', '2025-10-05', '2025-11-01', '2025-12-01', '2025-12-08', '2025-12-25',
    # 2026
    '2026-01-01', '2026-04-03', '2026-04-05', '2026-04-25',
    '2026-05-01', '2026-06-04', '2026-06-10', '2026-08-15',
    '2026-10-05', '2026-11-01', '2026-12-01', '2026-12-08', '2026-12-25'
])

df['feriado'] = df['data'].isin(feriados_2026).map({True: 'Y', False: 'N'})

# Pontes
def is_ponte(row):
    data, dia_semana = row['data'], row['dia_semana']
    if dia_semana == 'Thursday' and (data + pd.Timedelta(days=1)) in feriados_2026:
        return 'Y'
    if dia_semana == 'Friday' and (data + pd.Timedelta(days=3)) in feriados_2026:
        return 'Y'
    if dia_semana == 'Monday' and (data - pd.Timedelta(days=3)) in feriados_2026:
        return 'Y'
    return 'N'

df['ponte'] = df.apply(is_ponte, axis=1)

# Período ordenado (25 a 5)
df['periodo_ordenado'] = df['data'].dt.day.apply(
    lambda dia: 'Y' if (dia >= 25 or dia <= 5) else 'N'
)

# Jogos importantes (usar o calendário já definido na geração)
df['jogo_futebol'] = df['data'].dt.strftime('%Y-%m-%d').map(jogos_futebol_calendario).fillna('N')
df['tem_jogo'] = df['jogo_futebol'].apply(lambda x: 'Y' if x != 'N' else 'N')

# Mês e estação do ano
df['mes'] = df['data'].dt.month
df['estacao'] = df['mes'].apply(lambda m: 
    'Verão' if m in [6,7,8] else 
    'Outono' if m in [9,10,11] else 
    'Inverno' if m in [12,1,2] else 'Primavera'
)

print(f"   ✓ Feriados: {(df['feriado'] == 'Y').sum()} dias")
print(f"   ✓ Pontes: {(df['ponte'] == 'Y').sum()} dias")
print(f"   ✓ Período Ordenado: {(df['periodo_ordenado'] == 'Y').sum()} dias")
print(f"   ✓ Jogos Futebol: {(df['tem_jogo'] == 'Y').sum()} jogos")
print(f"   ✓ Dias de Chuva: {(df['choveu'] == 'Y').sum()} dias")
print(f"   ✓ Temperatura: {df['temperatura'].min()}°C - {df['temperatura'].max()}°C")

# Reorganizar colunas
cols = df.columns.tolist()
base_cols = ['data', 'dia_semana', 'clientes']
new_features = ['feriado', 'ponte', 'periodo_ordenado', 'tem_jogo', 'jogo_futebol', 
                'choveu', 'temperatura', 'mes', 'estacao']
other_cols = [col for col in cols if col not in base_cols + new_features]

df = df[base_cols + new_features + other_cols]

# Converter data de volta para string
df['data'] = df['data'].dt.strftime('%Y-%m-%d')

# Salvar CSV atualizado
df.to_csv('historico_restaurante.csv', index=False)

print("\n" + "=" * 80)
print("✅ DATASET DE 6 MESES GERADO COM SUCESSO!")
print("=" * 80)
print(f"\nPeríodo: {df['data'].min()} a {df['data'].max()}")
print(f"Total: {len(df)} dias de dados")
print("\nFeatures adicionadas:")
print("   • Feriados e pontes")
print("   • Período de ordenado (25-5)")
print("   • Jogos de futebol importantes (25 jogos)")
print("   • Clima: Chuva + Temperatura")
print("   • Sazonalidade: Mês + Estação do ano")
print("\n💡 Use compare_models.py para treinar e comparar modelos!")
print("=" * 80)
