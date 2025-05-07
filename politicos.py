import pandas as pd
from faker import Faker
import numpy as np
import random
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial
sns.set(style="whitegrid")
fake = Faker('es_AR')

# 1. Cargar datos
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_politica.csv"
df_politica = pd.read_csv(url)

# 2. Inserción de casos sospechosos simulados
num_politicos = len(df_politica)
num_sospechosos = int(num_politicos * 0.15)
sospechoso_indices = np.random.choice(df_politica.index, num_sospechosos, replace=False)
df_politica['Es_Sospechoso'] = 0
df_politica.loc[sospechoso_indices, 'Es_Sospechoso'] = 1

for idx in sospechoso_indices:
    if random.random() < 0.4:
        df_politica.loc[idx, 'Incremento_Patrimonial_Ultimo_Año'] = np.random.uniform(150000, 500000)
    elif random.random() < 0.3:
        df_politica.loc[idx, 'Donaciones_Recibidas'] = np.random.uniform(100000, 300000)
        df_politica.loc[idx, 'Gastos_Campania'] = np.random.uniform(5000, 50000)
    elif random.random() < 0.3:
        df_politica.loc[idx, 'Vinculos_Empresariales'] = 'No Declarados'
    if random.random() < 0.2:
        df_politica.loc[idx, 'Denuncias_Previas'] = random.randint(2, 5)
    if random.random() < 0.1 and df_politica.loc[idx, 'Cargo'] in ['Ministro', 'Senador']:
        df_politica.loc[idx, 'Incremento_Patrimonial_Ultimo_Año'] *= random.uniform(2, 4)

# 3. Ingeniería de características
df_politica['Ratio_Incremento_Patrimonio'] = df_politica['Incremento_Patrimonial_Ultimo_Año'] / (df_politica['Declaracion_Patrimonial_Ultimo_Año'] + 1e-6)
df_politica['Ratio_Gastos_Donaciones'] = df_politica['Gastos_Campania'] / (df_politica['Donaciones_Recibidas'] + 1e-6)
df_politica['Incremento_Alto'] = (df_politica['Incremento_Patrimonial_Ultimo_Año'] > 100000).astype(int)
df_politica['Muchas_Denuncias'] = (df_politica['Denuncias_Previas'] > 1).astype(int)

def label_encode(series):
    unique_vals = series.dropna().unique()
    val_dict = {val: i for i, val in enumerate(unique_vals)}
    return series.map(val_dict).fillna(-1).astype(int)

df_politica['Cargo_Cod'] = label_encode(df_politica['Cargo'])
df_politica['Partido_Politico_Cod'] = label_encode(df_politica['Partido_Politico'])
df_politica['Vinculos_Empresariales_Cod'] = label_encode(df_politica['Vinculos_Empresariales'])

# 4. Preparar datos
features = ['Declaracion_Patrimonial_Ultimo_Año', 'Incremento_Patrimonial_Ultimo_Año',
            'Donaciones_Recibidas', 'Gastos_Campania', 'Ratio_Incremento_Patrimonio',
            'Ratio_Gastos_Donaciones', 'Incremento_Alto', 'Muchas_Denuncias',
            'Cargo_Cod', 'Partido_Politico_Cod', 'Vinculos_Empresariales_Cod']
X = df_politica[features].fillna(0).values
y = df_politica['Es_Sospechoso'].values

# 5. División manual 70/30
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 6. Escalado manual
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-6
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# 7. Modelo con XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 8. Evaluación
accuracy = (y_pred == y_test).mean()
print("\nPrecisión del Modelo:", accuracy)

# Matriz de confusión
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predicción'], margins=True)
print("\nMatriz de Confusión:\n", conf_matrix)

# Reporte de clasificación básico
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))
precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
print("\nReporte de Clasificación:")
print(f"Precisión (Sospechoso): {precision:.2f}")
print(f"Recall (Sospechoso): {recall:.2f}")
print(f"F1-score (Sospechoso): {f1:.2f}")

# 9. Políticos sospechosos detectados
df_test = df_politica.iloc[test_idx].copy()
df_test['Prediccion_Sospechoso'] = y_pred
politicos_sospechosos = df_test[df_test['Prediccion_Sospechoso'] == 1][
    ['ID_Politico', 'Nombre_Apellido', 'Cargo', 'Partido_Politico',
     'Declaracion_Patrimonial_Ultimo_Año', 'Incremento_Patrimonial_Ultimo_Año',
     'Donaciones_Recibidas', 'Gastos_Campania', 'Vinculos_Empresariales',
     'Denuncias_Previas', 'Es_Sospechoso', 'Prediccion_Sospechoso']]
print("\nPolíticos Detectados como Potencialmente Sospechosos:")
print(politicos_sospechosos)

# 10. Importancia de características
importancias = model.feature_importances_
df_importancias = pd.DataFrame({'Caracteristica': features, 'Importancia': importancias})
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)
print("\nImportancia de las Características:")
print(df_importancias)

# 11. GRÁFICOS

# A. Distribución de sospechosos
plt.figure(figsize=(6, 4))
sns.countplot(x='Es_Sospechoso', data=df_politica, palette='coolwarm')
plt.title('Distribución de Casos Sospechosos')
plt.xticks([0, 1], ['No Sospechoso', 'Sospechoso'])
plt.xlabel('')
plt.ylabel('Cantidad')
plt.tight_layout()
plt.show()

# B. Importancia de características
plt.figure(figsize=(10, 6))
sns.barplot(data=df_importancias, x='Importancia', y='Caracteristica', palette='viridis')
plt.title('Importancia de Características según el Modelo')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.tight_layout()
plt.show()

# C. Matriz de Confusión
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix.iloc[:2, :2], annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.tight_layout()
plt.show()

# D. Distribución del incremento patrimonial
plt.figure(figsize=(8, 5))
sns.histplot(data=df_politica, x='Incremento_Patrimonial_Ultimo_Año', hue='Es_Sospechoso', bins=30, kde=True, palette='Set2')
plt.title('Distribución del Incremento Patrimonial por Tipo')
plt.xlabel('Incremento Patrimonial')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()
