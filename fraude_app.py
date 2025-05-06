import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
import plotly.express as px
import plotly.figure_factory as ff

# Título de la aplicación Streamlit
st.title("Detección de Fraude Interno en Rendiciones de Cuenta")
st.markdown("Esta aplicación simula la detección de posibles fraudes en rendiciones de cuenta internas utilizando un modelo de Machine Learning.")

# --- 1. Simulación de Datos ---
st.subheader("Simulación de Datos de Rendiciones de Cuenta")
num_rendiciones = st.sidebar.slider("Número de Rendiciones a Simular", 100, 500, 250)
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_rendiciones1.csv"
data_rendiciones = pd.read_csv(url)
df_rendiciones = pd.DataFrame(data_rendiciones)
df_rendiciones["Fecha_Presentacion"] = pd.to_datetime(
    df_rendiciones["Fecha_Presentacion"]
)

# Simulación de casos sospechosos con control deslizante
porcentaje_sospechosos = st.sidebar.slider("Porcentaje de Rendiciones Sospechosas", 1, 20, 7) / 100.0
num_sospechosos = int(num_rendiciones * porcentaje_sospechosos)
sospechosos_indices = np.random.choice(
    df_rendiciones.index, num_sospechosos, replace=False
)
df_rendiciones["Es_Sospechoso"] = 0
df_rendiciones.loc[sospechosos_indices, "Es_Sospechoso"] = 1

with st.expander("Mostrar Datos de Rendiciones Simulados"):
    st.dataframe(df_rendiciones.head())

# --- 2. Simulación de Comportamientos Sospechosos ---
st.subheader("Simulación de Comportamientos Sospechosos")
st.markdown("Se simulan algunos patrones que podrían indicar fraude.")

random_seed = st.sidebar.number_input("Semilla Aleatoria para Simulación", 1, 100, 42)
random.seed(random_seed)
np.random.seed(random_seed)
Faker.seed(random_seed)

for idx in sospechosos_indices:
    if random.random() < 0.4:
        fecha_base = df_rendiciones.loc[idx, "Fecha_Presentacion"]
        if idx + 1 < len(df_rendiciones):
            df_rendiciones.loc[
                idx + 1, "Fecha_Presentacion"
            ] = fecha_base + timedelta(days=random.randint(0, 5))
            df_rendiciones.loc[idx + 1, "Empleado"] = df_rendiciones.loc[
                idx, "Empleado"
            ]
            df_rendiciones.loc[idx + 1, "Tipo_Gasto"] = df_rendiciones.loc[
                idx, "Tipo_Gasto"
            ]
            df_rendiciones.loc[idx + 1, "Monto"] = (
                df_rendiciones.loc[idx, "Monto"] * random.uniform(0.8, 1.2)
            )
            df_rendiciones.loc[idx + 1, "Es_Sospechoso"] = 1
    elif random.random() < 0.3:
        df_rendiciones.loc[idx, "Tipo_Gasto"] = "Otros"
        df_rendiciones.loc[idx, "Monto"] = np.random.uniform(800, 2000)
    elif random.random() < 0.3:
        df_rendiciones.loc[idx, "Justificante_Adjunto"] = "No"
        df_rendiciones.loc[idx, "Monto"] = np.random.uniform(300, 700)
    if random.random() < 0.2:
        df_rendiciones.loc[idx, "Tipo_Gasto"] = "Viáticos"
        df_rendiciones.loc[idx, "Monto"] = np.random.uniform(400, 1000)
        fecha = df_rendiciones.loc[idx, "Fecha_Presentacion"]
        if fecha.weekday() >= 4:
            df_rendiciones.loc[idx, "Fecha_Presentacion"] = fecha - timedelta(
                days=random.randint(0, 2)
            )

# --- 3. Ingeniería de Características ---
st.subheader("Ingeniería de Características")
st.markdown("Se crean nuevas características para ayudar al modelo a identificar patrones sospechosos.")

df_rendiciones["Monto_Promedio_Tipo"] = df_rendiciones.groupby(
    "Tipo_Gasto"
)["Monto"].transform("mean")
df_rendiciones["Monto_Alto_Relativo"] = np.where(
    df_rendiciones["Monto"] > df_rendiciones["Monto_Promedio_Tipo"] * 2.5, 1, 0
)

umbral_justificante = st.sidebar.number_input("Umbral de Monto sin Justificante (Alto)", 50, 500, 200)
df_rendiciones["Sin_Justificante_Alto_Monto"] = np.where(
    (df_rendiciones["Justificante_Adjunto"] == "No")
    & (df_rendiciones["Monto"] > umbral_justificante),
    1,
    0,
)

umbral_otros_monto = st.sidebar.number_input("Umbral de Monto 'Otros' (Elevado)", 100, 1000, 500)
df_rendiciones["Otros_Monto_Elevado"] = np.where(
    (df_rendiciones["Tipo_Gasto"] == "Otros")
    & (df_rendiciones["Monto"] > umbral_otros_monto),
    1,
    0,
)

df_rendiciones["Fecha_Truncada"] = df_rendiciones["Fecha_Presentacion"].dt.date
frecuencia_empleado = (
    df_rendiciones.groupby(["Empleado", "Fecha_Truncada"])
    .size()
    .reset_index(name="Frecuencia")
)

df_rendiciones = pd.merge(
    df_rendiciones,
    frecuencia_empleado,
    on=["Empleado", "Fecha_Truncada"],
    how="left",
)

if "Frecuencia" in df_rendiciones.columns:
    umbral_frecuencia = st.sidebar.number_input("Umbral de Frecuencia de Gastos por Día", 1, 5, 3)
    df_rendiciones["Gastos_Frecuentes"] = (
        df_rendiciones["Frecuencia"] > umbral_frecuencia
    ).fillna(0).astype(int)
else:
    st.error("Error: La columna 'Frecuencia' no se creó correctamente durante la fusión.")

df_rendiciones["Dia_Semana"] = df_rendiciones[
    "Fecha_Presentacion"
].dt.dayofweek
df_rendiciones["Viaticos_Fin_Semana"] = np.where(
    (df_rendiciones["Tipo_Gasto"] == "Viáticos")
    & (df_rendiciones["Dia_Semana"] >= 4),
    1,
    0,
)

palabras_genericas = ['pago', 'gasto', 'servicio']
df_rendiciones['Descripcion_Generica'] = df_rendiciones['Descripcion'].apply(
    lambda x: 1 if any(palabra in x.lower() for palabra in palabras_genericas) else 0
)
umbral_descripcion_generica_monto = st.sidebar.number_input("Umbral de Monto con Descripción Genérica (Alto)", 100, 800, 300)
df_rendiciones["Descripcion_Generica_Monto_Alto"] = np.where(
    (df_rendiciones["Descripcion_Generica"] == 1)
    & (df_rendiciones["Monto"] > umbral_descripcion_generica_monto),
    1,
    0,
)

with st.expander("Mostrar DataFrame con Características de Ingeniería"):
    st.dataframe(df_rendiciones.head())

# --- 4. Preparación de Datos para el Modelo ---
st.subheader("Preparación de Datos para el Modelo")
features = [
    "Monto",
    "Monto_Alto_Relativo",
    "Sin_Justificante_Alto_Monto",
    "Otros_Monto_Elevado",
    "Gastos_Frecuentes",
    "Viaticos_Fin_Semana",
    "Descripcion_Generica_Monto_Alto",
    "Monto_Promedio_Tipo",
]
X = df_rendiciones[features]
y = df_rendiciones["Es_Sospechoso"]
X = X.fillna(0)

# --- 5. Diseño y Evaluación del Modelo de Detección de Fraude ---
st.subheader("Modelo de Detección de Fraude Interno")
st.markdown("Se entrena un modelo de Random Forest para detectar posibles fraudes.")

random_state = st.sidebar.number_input("Semilla Aleatoria para el Modelo", 1, 100, 42)
modelo_deteccion_fraude_interno = RandomForestClassifier(random_state=random_state)

# Validación cruzada
n_splits = st.sidebar.slider("Número de Folds para Validación Cruzada", 2, 10, 5)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
scores = cross_val_score(
    modelo_deteccion_fraude_interno, X, y, cv=cv, scoring="accuracy"
)
st.write(f"Precisión de Validación Cruzada (Promedio): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

# Entrenamiento y Evaluación en el conjunto de prueba
test_size = st.sidebar.slider("Tamaño del Conjunto de Prueba (%)", 10, 50, 30) / 100.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo_deteccion_fraude_interno.fit(X_train_scaled, y_train)
y_pred_fraude_interno = modelo_deteccion_fraude_interno.predict(
    X_test_scaled
)

st.subheader("Evaluación del Modelo")
st.write(f"Precisión en el Conjunto de Prueba: {accuracy_score(y_test, y_pred_fraude_interno):.4f}")
st.markdown("Reporte de Clasificación:")
st.text(classification_report(
    y_test,
    y_pred_fraude_interno,
    target_names=["No Sospechoso", "Sospechoso"],
))

cm = confusion_matrix(y_test, y_pred_fraude_interno)
fig_cm = ff.create_annotated_heatmap(cm, x=["No Sospechoso", "Sospechoso"], y=["No Sospechoso", "Sospechoso"], colorscale='Blues')
fig_cm.update_layout(title_text='Matriz de Confusión', title_x=0.5)
st.plotly_chart(fig_cm)

# --- 6. Identificación de Rendiciones Sospechosas ---
st.subheader("Identificación de Rendiciones Sospechosas")
df_test = df_rendiciones.loc[X_test.index].copy()
df_test['Prediccion_Sospechoso'] = y_pred_fraude_interno
rendiciones_sospechosas = df_test[df_test['Prediccion_Sospechoso'] == 1][
    ['ID_Rendicion', 'Fecha_Presentacion', 'Empleado', 'Departamento', 'Tipo_Gasto', 'Monto',
     'Justificante_Adjunto', 'Estado_Aprobacion', 'Es_Sospechoso', 'Prediccion_Sospechoso']]

if not rendiciones_sospechosas.empty:
    st.markdown("### Rendiciones de Cuenta Marcadas como Sospechosas:")
    st.dataframe(rendiciones_sospechosas)
    st.warning("Estas rendiciones podrían requerir una revisión más exhaustiva.")
else:
    st.info("El sistema no detectó rendiciones sospechosas en el conjunto de prueba.")

# --- 7. Análisis de Importancia de Características ---
st.subheader("Importancia de las Características")
if hasattr(modelo_deteccion_fraude_interno, 'feature_importances_'):
    importancia_caracteristicas = pd.DataFrame(
        {'Caracteristica': features, 'Importancia': modelo_deteccion_fraude_interno.feature_importances_})
    importancia_caracteristicas = importancia_caracteristicas.sort_values(by='Importancia', ascending=False)

    fig_importance = px.bar(importancia_caracteristicas, x='Importancia', y='Caracteristica',
                             title='Importancia de las Características en el Modelo',
                             labels={'Importancia': 'Puntaje de Importancia', 'Caracteristica': 'Característica'})
    st.plotly_chart(fig_importance)
else:
    st.warning("El modelo no tiene la propiedad 'feature_importances_'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desarrollado con Streamlit.")
