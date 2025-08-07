# ===============================================================
# SCRIPT DE AUDITORÍA DE COLOCACIONES CON STREAMLIT Y DOCKER
# ===============================================================

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

st.set_page_config (page_title="Auditoría de Colocaciones", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos de colocaciones simulados para la auditoría."""
    np.random.seed (123)
    random.seed (123)
    fake = Faker ('es_AR')
    Faker.seed (123)
    num_colocaciones = 150
    tipos_colocacion = ['Préstamo', 'Inversión']
    tipos_interes = ['Fijo', 'Variable']
    estados = ['Vigente', 'Cancelado', 'Mora']

    colocaciones = []
    for i in range (num_colocaciones):
        tipo = random.choice (tipos_colocacion)
        monto = round (random.uniform (50000, 1000000), 2)
        tasa_interes = round (random.uniform (5, 30), 2)
        tipo_tasa = random.choice (tipos_interes)
        plazo_meses = random.choice ([6, 12, 24, 36, 48, 60])
        fecha_inicio = fake.date_between (start_date='-5y', end_date='-1y')
        fecha_vencimiento = fecha_inicio + timedelta (days=plazo_meses * 30)
        estado = random.choices (estados, weights=[0.7, 0.2, 0.1])[0]

        colocaciones.append ({
            'id_colocacion': f'CL-{1000 + i}',
            'cliente_id': fake.random_int (min=1000, max=9999),
            'tipo_colocacion': tipo,
            'monto': monto,
            'tasa_interes': tasa_interes,
            'tipo_tasa': tipo_tasa,
            'plazo_meses': plazo_meses,
            'fecha_inicio': fecha_inicio,
            'fecha_vencimiento': fecha_vencimiento,
            'estado': estado
        })

    return pd.DataFrame (colocaciones)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica la detección de anomalías y las reglas heurísticas."""
    df['fecha_inicio'] = pd.to_datetime (df['fecha_inicio'], errors='coerce')
    df['fecha_vencimiento'] = pd.to_datetime (df['fecha_vencimiento'], errors='coerce')

    # Detección de Anomalías (Isolation Forest)
    features = ['monto', 'tasa_interes', 'plazo_meses']
    X = df[features].copy ()
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X)
    modelo = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = modelo.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    # Reglas Heurísticas
    def alerta_reglas(row):
        alertas = []
        hoy = pd.to_datetime ("today")
        if row['estado'] == 'Mora':
            alertas.append ("En mora")
        if row['tasa_interes'] > 25:
            alertas.append ("Tasa alta")
        if row['tasa_interes'] < 2:
            alertas.append ("Tasa baja")
        if pd.notnull (row['fecha_vencimiento']) and row['fecha_vencimiento'] < hoy and row['estado'] != 'Cancelado':
            alertas.append ("Vencida no cancelada")
        return " | ".join (alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply (alerta_reglas, axis=1)

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("🏦 Auditoría de Colocaciones")
st.markdown (
    "Esta aplicación realiza una auditoría de colocaciones simuladas, identificando anomalías con un modelo de **Isolation Forest** y aplicando reglas heurísticas.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_colocaciones = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_colocaciones)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total de Colocaciones", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)
        with col3:
            alertas_count = len (df_auditado[df_auditado['alerta_heuristica'] != "Sin alertas"])
            st.metric ("Colocaciones con Alertas", alertas_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "Sin alertas")]

        st.subheader ("Colocaciones Anómalas o con Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_colocacion', 'cliente_id', 'tipo_colocacion', 'monto', 'tasa_interes',
                                'plazo_meses', 'estado', 'resultado_auditoria', 'alerta_heuristica']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_colocaciones.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron colocaciones anómalas o con alertas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Distribución de Tasa de Interés
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        sns.histplot (df_auditado['tasa_interes'], bins=15, kde=True, color='skyblue', ax=ax1)
        ax1.set_title ('Distribución de Tasa de Interés')
        ax1.set_xlabel ('Tasa (%)')
        ax1.set_ylabel ('Frecuencia')
        st.pyplot (fig1)

        # Gráfico 2: Anomalias en Monto vs Tasa
        fig2, ax2 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            data=df_auditado, x='monto', y='tasa_interes',
            hue='resultado_auditoria', style='estado', size='plazo_meses',
            sizes=(50, 250), palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.7, ax=ax2
        )
        ax2.set_title ('Monto vs Tasa de Interés (Detección de Anomalías)')
        ax2.set_xlabel ('Monto ($)')
        ax2.set_ylabel ('Tasa de Interés (%)')
        ax2.get_xaxis ().set_major_formatter (plt.FuncFormatter (lambda x, p: format (int (x), ',')))
        ax2.legend (title='Resultado Auditoría')
        st.pyplot (fig2)