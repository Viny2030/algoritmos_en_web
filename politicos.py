import streamlit as st
import pandas as pd
import numpy as np
import random
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("🔍 Detección de Corrupción en Política (sin scikit-learn)")

# 1. Cargar datos
fake = Faker('es_AR')
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_politica.csv"
df_politica = pd.read_csv(url)

# 2. Simulación de casos sospechosos
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
    if random.random() < 0.1:
        if df_politica.loc[idx, 'Cargo'] in ['Ministro', 'Senador']:
            df_politica.loc[idx, 'Incremento_Patrimonial_Ultimo_Año'] *= random.uniform(2, 4)

# 3. Ingeniería de características
df_politica['Ratio_Incremento_Patrimonio'] = df_politica['Incremento_Patrimonial_Ultimo_Año'] / (
    df_politica['Declaracion_Patrimonial_Ultimo_Año'] + 1e-6)
df_politica['Ratio_Gastos_Donaciones'] = df_politica['Gastos_Campania'] / (
    df_politica['Donaciones_Recibidas'] + 1e-6)
df_politica['Incremento_Alto'] = (df_politica['Incremento_Patrimonial_Ultimo_Año'] > 100000).astype(int)
df_politica['Muchas_Denuncias'] = (df_politica['Denuncias_Previas'] > 1).astype(int)
df_politica['Cargo_Cod'] = pd.factorize(df_politica['Cargo'])[0]
df_politica['Partido_Politico_Cod'] = pd.factorize(df_politica['Partido_Politico'])[0]
df_politica['Vinculos_Empresariales_Cod'] = pd.factorize(df_politica['Vinculos_Empresariales'])[0]

# 4. Clasificador manual
def clasificador_manual(fila):
    if fila['Incremento_Alto'] == 1 and fila['Muchas_Denuncias'] == 1:
        return 1
    if fila['Ratio_Incremento_Patrimonio'] > 1.5 and fila['Vinculos_Empresariales_Cod'] == 1:
        return 1
    return 0

df_politica['Prediccion_Sospechoso'] = df_politica.apply(clasificador_manual, axis=1)

# 5. Evaluación
def evaluar(y_real, y_pred):
    vp = sum((y_real == 1) & (y_pred == 1))
    vn = sum((y_real == 0) & (y_pred == 0))
    fp = sum((y_real == 0) & (y_pred == 1))
    fn = sum((y_real == 1) & (y_pred == 0))
    precision = (vp + vn) / len(y_real)
    return precision, vp, fp, fn, vn

precision, vp, fp, fn, vn = evaluar(df_politica['Es_Sospechoso'], df_politica['Prediccion_Sospechoso'])

st.subheader("📊 Evaluación del Clasificador")
st.write(f"**Precisión total:** {precision:.2f}")
st.write(f"**VP:** {vp} | **FP:** {fp} | **FN:** {fn} | **VN:** {vn}")

# 6. Visualizaciones
sns.set(style="whitegrid")

st.subheader("📌 Visualizaciones")

# a) Sospechosos por cargo
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.countplot(data=df_politica, x='Cargo', hue='Prediccion_Sospechoso', ax=ax1)
plt.title("Sospechosos Detectados por Cargo")
plt.xticks(rotation=45)
st.pyplot(fig1)

# b) Ratio de incremento patrimonial
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_politica, x='Es_Sospechoso', y='Ratio_Incremento_Patrimonio', ax=ax2)
plt.title("Ratio Incremento Patrimonial vs. Sospechosos")
st.pyplot(fig2)

# c) Mapa de correlaciones
fig3, ax3 = plt.subplots(figsize=(10, 8))
features_corr = ['Declaracion_Patrimonial_Ultimo_Año', 'Incremento_Patrimonial_Ultimo_Año',
                 'Donaciones_Recibidas', 'Gastos_Campania', 'Ratio_Incremento_Patrimonio',
                 'Ratio_Gastos_Donaciones', 'Incremento_Alto', 'Muchas_Denuncias',
                 'Cargo_Cod', 'Partido_Politico_Cod', 'Vinculos_Empresariales_Cod', 'Es_Sospechoso']
sns.heatmap(df_politica[features_corr].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
plt.title("Mapa de Correlaciones")
st.pyplot(fig3)

# 7. Resultados
st.subheader("🕵️ Políticos Detectados como Sospechosos")
sospechosos = df_politica[df_politica['Prediccion_Sospechoso'] == 1][
    ['ID_Politico', 'Nombre_Apellido', 'Cargo', 'Partido_Politico',
     'Declaracion_Patrimonial_Ultimo_Año', 'Incremento_Patrimonial_Ultimo_Año',
     'Donaciones_Recibidas', 'Gastos_Campania', 'Vinculos_Empresariales',
     'Denuncias_Previas', 'Es_Sospechoso', 'Prediccion_Sospechoso']
]
st.dataframe(sospechosos)

