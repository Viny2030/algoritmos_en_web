# =================================================================
# SCRIPT DE AUDITORÍA DE PRODUCTOS EN PROCESO CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# ===============================================================

st.set_page_config (page_title="Auditoría de Productos en Proceso", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de productos en proceso."""
    np.random.seed (42)
    random.seed (42)
    fake = Faker ('es_AR')
    Faker.seed (42)
    num_registros = 50
    etapas_produccion = ['Corte', 'Ensamblado', 'Soldadura', 'Pintura', 'Control de calidad']
    lineas_produccion = ['Línea A', 'Línea B', 'Línea C']
    productos = ['Silla', 'Mesa', 'Estantería', 'Armario', 'Puerta']

    productos_proceso = []
    for i in range (num_registros):
        fecha_inicio = fake.date_between (start_date='-30d', end_date='-1d')
        duracion_estimada = random.randint (1, 10)
        avance_porcentaje = random.randint (10, 95)

        productos_proceso.append ({
            'id_proceso': f'WIP-{1000 + i}',
            'producto': random.choice (productos),
            'lote': f'L-{fake.random_int (min=1000, max=9999)}',
            'etapa_actual': random.choice (etapas_produccion),
            'linea_produccion': random.choice (lineas_produccion),
            'cantidad_en_proceso': random.randint (10, 200),
            'fecha_inicio': fecha_inicio,
            'fecha_estim_termino': fecha_inicio + timedelta (days=duracion_estimada),
            'avance_porcentaje': avance_porcentaje,
            'estado': 'En proceso'
        })

    return pd.DataFrame (productos_proceso)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_inicio'] = pd.to_datetime (df['fecha_inicio'])
    df['fecha_estim_termino'] = pd.to_datetime (df['fecha_estim_termino'])

    def reglas_auditoria(row):
        alertas = []
        hoy = pd.to_datetime ("today")
        if row['avance_porcentaje'] <= 10 and row['estado'] == 'En proceso':
            alertas.append ("Avance muy bajo")
        if row['fecha_estim_termino'] < hoy and row['estado'] == 'En proceso':
            alertas.append ("Fecha estimada vencida")
        if row['cantidad_en_proceso'] <= 0:
            alertas.append ("Cantidad inválida")
        if row['avance_porcentaje'] > 100 or row['avance_porcentaje'] < 0:
            alertas.append ("Avance fuera de rango")
        return " | ".join (alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply (reglas_auditoria, axis=1)

    features = ['cantidad_en_proceso', 'avance_porcentaje']
    X = df[features].fillna (0)
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X)
    modelo = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = modelo.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("⚙️ Auditoría de Productos en Proceso (WIP)")
st.markdown (
    "Esta aplicación audita datos simulados de productos en proceso, identificando anomalías y aplicando reglas de negocio.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_proceso = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_proceso)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Procesos", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "Sin alertas")]

        st.subheader ("Procesos Anómalos o con Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_proceso', 'producto', 'etapa_actual', 'linea_produccion', 'cantidad_en_proceso',
                                'avance_porcentaje', 'estado', 'alerta_heuristica', 'resultado_auditoria']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_wip.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías o alertas significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Avance por Etapa
        fig1, ax1 = plt.subplots (figsize=(12, 7))
        sns.boxplot (data=df_auditado, x='avance_porcentaje', y='etapa_actual', hue='resultado_auditoria',
                     palette={'Normal': 'skyblue', 'Anómalo': 'salmon'}, ax=ax1)
        ax1.set_title ('Avance por Etapa de Producción')
        ax1.set_xlabel ('Avance (%)')
        ax1.set_ylabel ('Etapa Actual')
        st.pyplot (fig1)

        # Gráfico 2: Cantidad en Proceso vs Avance
        fig2, ax2 = plt.subplots (figsize=(10, 6))
        sns.scatterplot (data=df_auditado, x='cantidad_en_proceso', y='avance_porcentaje', hue='resultado_auditoria',
                         style='linea_produccion', palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.8, s=100,
                         ax=ax2)
        ax2.set_title ('Cantidad en Proceso vs Avance (%)')
        ax2.set_xlabel ('Cantidad en Proceso (unidades)')
        ax2.set_ylabel ('Avance (%)')
        st.pyplot (fig2)