import streamlit as st
import pandas as pd
from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import random
import plotly.express as px
import plotly.figure_factory as ff

# Título de la aplicación Streamlit
st.title("Detección de Potencial Corrupción Política")
st.markdown("Esta aplicación simula la detección de posibles casos de corrupción en el ámbito político utilizando un modelo de Machine Learning.")

# --- 1. Generación de un dataset simulado ---
st.subheader("Simulación de Datos de Políticos")
num_politicos_slider = st.sidebar.slider("Número de Políticos a Simular", 50, 200, 55)
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_politica.csv"
df_politica = pd.read_csv(url)

with st.expander("Mostrar Datos de Políticos Simulados"):
    st.dataframe(df_politica.head())

# --- 2. Inserción de Casos Sospechosos Simulados ---
st.subheader("Simulación de Casos Sospechosos")
porcentaje_sospechosos_slider = st.sidebar.slider("Porcentaje de Políticos Sospechosos", 5, 30, 15) / 100.0
num_sospechosos = int(num_politicos_slider * porcentaje_sospechosos_slider)
sospechoso_indices = np.random.choice(df_politica.index, num_sospechosos, replace=False)
df_politica['Es_Sospechoso'] = 0
df_politica.loc[sospechoso_indices, 'Es_Sospechoso'] = 1

st.markdown("Simulando patrones sospechosos...")
random_seed_sim = st.sidebar.number_input("Semilla Aleatoria para Simulación", 1, 100, 42)
random.seed(random_seed_sim)
np.random.seed(random_seed_sim)

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

with st.expander("Mostrar Datos con Casos Sospechosos Simulados"):
    st.dataframe(df_politica.head())

# --- 3. Ingeniería de Características ---
st.subheader("Ingeniería de Características")
st.markdown("Creando nuevas características para ayudar al modelo a identificar patrones.")

df_politica['Ratio_Incremento_Patrimonio'] = df_politica['Incremento_Patrimonial_Ultimo_Año'] / (df_politica['Declaracion_Patrimonial_Ultimo_Año'] + 1e-6)
df_politica['Ratio_Gastos_Donaciones'] = df_politica['Gastos_Campania'] / (df_politica['Donaciones_Recibidas'] + 1e-6)

umbral_alto_incremento_slider = st.sidebar.number_input("Umbral de Alto Incremento Patrimonial", 50000, 300000, 100000)
df_politica['Incremento_Alto'] = (df_politica['Incremento_Patrimonial_Ultimo_Año'] > umbral_alto_incremento_slider).astype(int)

umbral_denuncias_slider = st.sidebar.number_input("Umbral de Muchas Denuncias Previas", 0, 3, 1)
df_politica['Muchas_Denuncias'] = (df_politica['Denuncias_Previas'] > umbral_denuncias_slider).astype(int)

le_cargo = LabelEncoder()
df_politica['Cargo_Cod'] = le_cargo.fit_transform(df_politica['Cargo'])

le_partido = LabelEncoder()
df_politica['Partido_Politico_Cod'] = le_partido.fit_transform(df_politica['Partido_Politico'])

le_vinculos = LabelEncoder()
df_politica['Vinculos_Empresariales_Cod'] = le_vinculos.fit_transform(df_politica['Vinculos_Empresariales'])

with st.expander("Mostrar DataFrame con Ingeniería de Características"):
    st.dataframe(df_politica.head())

# --- 4. Selección de Características y Preparación de Datos ---
st.subheader("Preparación de Datos para el Modelo")
features = ['Declaracion_Patrimonial_Ultimo_Año', 'Incremento_Patrimonial_Ultimo_Año',
            'Donaciones_Recibidas', 'Gastos_Campania', 'Ratio_Incremento_Patrimonio',
            'Ratio_Gastos_Donaciones', 'Incremento_Alto', 'Muchas_Denuncias',
            'Cargo_Cod', 'Partido_Politico_Cod', 'Vinculos_Empresariales_Cod']
X = df_politica[features]
y = df_politica['Es_Sospechoso']
X = X.fillna(0)

# --- 5. División de Datos ---
st.subheader("División de Datos")
test_size_slider = st.sidebar.slider("Tamaño del Conjunto de Prueba (%)", 10, 50, 30) / 100.0
random_state_split = st.sidebar.number_input("Semilla Aleatoria para División de Datos", 1, 100, 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_slider, random_state=random_state_split)

st.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
st.write(f"Tamaño del conjunto de prueba: {X_test.shape[0]}")

# --- 6. Escalado de Características ---
st.subheader("Escalado de Características")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 7. Entrenamiento del Modelo ---
st.subheader("Entrenamiento del Modelo de Clasificación (Random Forest)")
random_state_model = st.sidebar.number_input("Semilla Aleatoria para el Modelo", 1, 100, 42)
model = RandomForestClassifier(random_state=random_state_model)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# --- 8. Evaluación del Modelo ---
st.subheader("Evaluación del Modelo")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Precisión del Modelo:** {accuracy:.4f}")

st.markdown("Reporte de Clasificación:")
report = classification_report(y_test, y_pred, target_names=['No Sospechoso', 'Sospechoso'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(cm, x=['No Sospechoso', 'Sospechoso'], y=['No Sospechoso', 'Sospechoso'], colorscale='Blues')
fig_cm.update_layout(title_text='Matriz de Confusión', title_x=0.5)
st.plotly_chart(fig_cm)

# --- 9. Políticos Detectados como Sospechosos ---
st.subheader("Políticos Detectados como Potencialmente Sospechosos")
df_test_with_predictions = df_politica.loc[X_test.index].copy()
df_test_with_predictions['Prediccion_Sospechoso'] = y_pred
politicos_sospechosos_pred = df_test_with_predictions[df_test_with_predictions['Prediccion_Sospechoso'] == 1][
    ['ID_Politico', 'Nombre_Apellido', 'Cargo', 'Partido_Politico', 'Declaracion_Patrimonial_Ultimo_Año',
     'Incremento_Patrimonial_Ultimo_Año', 'Donaciones_Recibidas', 'Gastos_Campania',
     'Vinculos_Empresariales', 'Denuncias_Previas', 'Es_Sospechoso', 'Prediccion_Sospechoso']]

if not politicos_sospechosos_pred.empty:
    st.dataframe(politicos_sospechosos_pred)
else:
    st.info("No se detectaron políticos sospechosos en el conjunto de prueba.")

# --- 10. Análisis de Importancia de Características ---
st.subheader("Importancia de las Características")
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({'Característica': features, 'Importancia': model.feature_importances_})
    importance_df = importance_df.sort_values(by='Importancia', ascending=False)
    fig_importance = px.bar(importance_df, x='Importancia', y='Característica', title='Importancia de las Características')
    st.plotly_chart(fig_importance)
else:
    st.warning("El modelo no tiene la propiedad 'feature_importances_'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado con Streamlit.")
