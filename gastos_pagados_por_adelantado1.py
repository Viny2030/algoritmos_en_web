# =================================================================
# SCRIPT DE AUDITORÍA DE GASTOS PREPAGOS CON STREAMLIT Y DOCKER
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

st.set_page_config (page_title="Auditoría de Gastos Prepagos", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de gastos prepagos para la auditoría."""
    np.random.seed (42)
    random.seed (42)
    fake = Faker ('es_AR')
    Faker.seed (42)

    num_registros = 30
    tipos_gasto = ['Alquiler', 'Seguro', 'Publicidad', 'Licencias de software', 'Mantenimiento']
    proveedores = [fake.company () for _ in range (10)]

    prepagos = []
    for i in range (num_registros):
        proveedor = random.choice (proveedores)
        tipo = random.choice (tipos_gasto)
        fecha_pago = fake.date_between (start_date='-60d', end_date='today')
        duracion_meses = random.choice ([1, 3, 6, 12])
        fecha_inicio = fecha_pago + timedelta (days=random.randint (0, 15))
        fecha_fin = fecha_inicio + pd.DateOffset (months=duracion_meses)
        monto_total = round (random.uniform (5000, 200000), 2)
        monto_mensual = round (monto_total / duracion_meses, 2)

        prepagos.append ({
            'id_prepago': f'PP-{1000 + i}',
            'tipo_gasto': tipo,
            'proveedor': proveedor,
            'fecha_pago': fecha_pago,
            'fecha_inicio_servicio': fecha_inicio,
            'fecha_fin_servicio': fecha_fin.date (),
            'duracion_meses': duracion_meses,
            'monto_total': monto_total,
            'monto_mensual': monto_mensual,
            'moneda': 'ARS',
            'estado': 'Activo' if fecha_fin.date () > datetime.today ().date () else 'Consumido'
        })
    return pd.DataFrame (prepagos)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_pago'] = pd.to_datetime (df['fecha_pago'], errors='coerce')
    df['fecha_inicio_servicio'] = pd.to_datetime (df['fecha_inicio_servicio'], errors='coerce')
    df['fecha_fin_servicio'] = pd.to_datetime (df['fecha_fin_servicio'], errors='coerce')

    def reglas_auditoria(row):
        alertas = []
        if row['monto_total'] <= 0: alertas.append ("Monto total inválido")
        if row['monto_mensual'] <= 0: alertas.append ("Monto mensual inválido")
        if row['fecha_inicio_servicio'] < row['fecha_pago']: alertas.append ("Servicio comenzó antes del pago")
        if row['fecha_fin_servicio'] <= row['fecha_inicio_servicio']: alertas.append ("Rango de fechas inconsistente")
        if not np.isclose (row['monto_mensual'] * row['duracion_meses'], row['monto_total']):
            alertas.append ("Montos inconsistentes (total vs mensual)")
        return " | ".join (alertas) if alertas else "OK"

    df['alerta_heuristica'] = df.apply (reglas_auditoria, axis=1)

    features = ['monto_total', 'monto_mensual', 'duracion_meses']
    X = df[features].fillna (0)
    X_scaled = StandardScaler ().fit_transform (X)
    model = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("💰 Auditoría de Gastos Pagados por Adelantado")
st.markdown (
    "Esta aplicación audita gastos prepagos simulados, identificando anomalías con **Isolation Forest** y aplicando reglas de negocio.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_prepagos = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_prepagos)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Registros", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "OK")]

        st.subheader ("Gastos con Anomalías o Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_prepago', 'tipo_gasto', 'proveedor', 'monto_total', 'alerta_heuristica',
                                'resultado_auditoria']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_prepagos.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías o alertas significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Distribución de Monto Mensual
        fig1, ax1 = plt.subplots (figsize=(8, 4))
        sns.histplot (df_auditado['monto_mensual'], bins=15, kde=True, color='skyblue', ax=ax1)
        ax1.set_title ("Distribución de Monto Mensual")
        st.pyplot (fig1)

        # Gráfico 2: Monto Total vs Monto Mensual
        fig2, ax2 = plt.subplots (figsize=(8, 6))
        sns.scatterplot (data=df_auditado, x='monto_mensual', y='monto_total', hue='resultado_auditoria',
                         palette={'Normal': 'blue', 'Anómalo': 'red'}, ax=ax2)
        ax2.set_title ("Monto Total vs Monto Mensual")
        st.pyplot (fig2)

        # Gráfico 3: Distribución de Monto Total por Tipo de Gasto
        fig3, ax3 = plt.subplots (figsize=(10, 5))
        sns.boxplot (data=df_auditado, x='tipo_gasto', y='monto_total', ax=ax3)
        ax3.set_title ("Distribución de Monto Total por Tipo de Gasto")
        ax3.tick_params (axis='x', rotation=45)
        st.pyplot (fig3)

        # Gráfico 4: Distribución de Montos por Proveedor (Top 5)
        top_proveedores = df_auditado['proveedor'].value_counts ().head (5).index
        fig4, ax4 = plt.subplots (figsize=(10, 5))
        sns.boxplot (data=df_auditado[df_auditado['proveedor'].isin (top_proveedores)], x='proveedor', y='monto_total',
                     ax=ax4)
        ax4.set_title ("Distribución de Montos por Proveedor (Top 5)")
        ax4.tick_params (axis='x', rotation=45)
        st.pyplot (fig4)
