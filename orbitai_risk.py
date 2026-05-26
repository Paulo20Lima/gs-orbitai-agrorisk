import os 
from datetime import datetime 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data_atual = datetime.now().strftime("%d/%m/%Y - %H:%M")

for pasta in ['data', 'img']:
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        print(f" Pasta '{pasta}' criada com sucesso!")

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

np.random.seed(42)
n_amostras = 1000

data = {
    'latitude': np.random.uniform(-30, -5, n_amostras),
    'longitude': np.random.uniform(-60, -40, n_amostras),
    'temperatura_media_C': np.random.uniform(18, 40, n_amostras),
    'precipitacao_anual_mm': np.random.uniform(400, 2500, n_amostras),
    'umidade_solo_pct': np.random.uniform(10, 90, n_amostras),
    'indice_NDVI': np.random.uniform(0.1, 0.9, n_amostras)
}

df = pd.DataFrame(data)

print("\n🔧 [Data Cleaning] Simulando falhas de leitura de satélite...")
df.loc[df.sample(frac=0.03).index, 'indice_NDVI'] = np.nan 
nulos_antes = df['indice_NDVI'].isnull().sum()
print(f"Leituras de NDVI corrompidas encontradas: {nulos_antes} registros.")

df['indice_NDVI'] = df['indice_NDVI'].fillna(df['indice_NDVI'].mean())
print(f"Tratamento concluído. Valores nulos restantes: {df['indice_NDVI'].isnull().sum()}")

score_risco = (df['temperatura_media_C'] * 0.3) - (df['precipitacao_anual_mm'] * 0.01) - (df['umidade_solo_pct'] * 0.4) - (df['indice_NDVI'] * 10)

def rotular_risco(score):
    if score > -15: return 2  
    elif score > -30: return 1  
    else: return 0  

df['risco_climatico'] = score_risco.apply(rotular_risco)

df.to_csv('data/base_orbitai_limpa.csv', index=False)
print("[Data Export] Base de dados gerada com sucesso: 'data/base_orbitai_limpa.csv'\n")

X = df.drop(columns=['risco_climatico'])
y = df['risco_climatico']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

modelo_orbitai = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_orbitai.fit(X_train, y_train)
y_pred = modelo_orbitai.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print("==========================================")
print(f" ACURÁCIA DO MODELO: {acuracia * 100:.2f}%")
print("==========================================\n")
print(classification_report(y_test, y_pred, target_names=['Baixo Risco', 'Médio Risco', 'Alto Risco']))


plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Baixo', 'Médio', 'Alto'], yticklabels=['Baixo', 'Médio', 'Alto'])


plt.title('Matriz de Confusão - OrbitAI AgroRisk', pad=15)
plt.suptitle(f'Gerado em: {data_atual}', fontsize=9, color='gray', y=0.94)

plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')
plt.tight_layout()
plt.savefig('img/matriz_confusao.png')
plt.close()


importancias = modelo_orbitai.feature_importances_
indices = np.argsort(importancias)[::-1]
plt.figure(figsize=(10, 5))
sns.barplot(x=importancias[indices], y=X.columns[indices], hue=X.columns[indices], palette='viridis', legend=False)


plt.title('Importância das Variáveis na Previsão de Risco', pad=15)
plt.suptitle(f'Gerado em: {data_atual}', fontsize=9, color='gray', y=0.95)

plt.tight_layout()
plt.savefig('img/importancia_variaveis.png')
plt.close()

print("[Gráficos] 'img/matriz_confusao.png' e 'img/importancia_variaveis.png' salvos com sucesso.")


def simular_risco_fazenda(lat, log, temp, chuva, umidade_solo, ndvi):
    nova_area = pd.DataFrame([{
        'latitude': lat, 'longitude': log, 'temperatura_media_C': temp,
        'precipitacao_anual_mm': chuva, 'umidade_solo_pct': umidade_solo, 'indice_NDVI': ndvi
    }])
    
    predicao = modelo_orbitai.predict(nova_area)[0]
    probabilidades = modelo_orbitai.predict_proba(nova_area)[0]
    
    classes = ['🟢 BAIXO RISCO', '🟡 MÉDIO RISCO', '🔴 ALTO RISCO (CRÍTICO)']
    
    print("\n==========================================")
    print("🔮 ORBITAI AGRORISK - DIAGNÓSTICO DE ÁREA")
    print("==========================================")
    print(f"📍 Coordenadas: {lat}, {log}")
    print(f"📊 Status Predito: {classes[predicao]}")
    print(f"📈 Confiança do Modelo: {probabilidades[predicao]*100:.2f}%")
    print("==========================================\n")

simular_risco_fazenda(
    lat=-22.1234, log=-48.5678, 
    temp=38.5, chuva=450.0, 
    umidade_solo=12.0, ndvi=0.15
)