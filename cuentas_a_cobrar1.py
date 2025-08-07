# ===============================================================
# SCRIPT DE AUDITORÍA DE CUENTAS A COBRAR CON STREAMLIT Y DOCKER
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

st.set_page_config (page_title="Auditoría de Cuentas a Cobrar", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos de cuentas a cobrar simulados para la auditoría."""
    np.random.seed (123)
    random.seed (123)
    fake = Faker ('es_AR')
    Faker.seed (123)

    num_cuentas = 150
    clientes = [{'cliente_id': 1000 + i, 'nombre_cliente': fake.company ()} for i in range (30)]
    estados_cuenta = ['Vigente', 'Vencida', 'Pagada']

    cuentas_a_cobrar = []
    for i in range (num_cuentas):
        cliente = random.choice (clientes)
        fecha_emision = fake.date_between (start_date='-2y', end_date='-1M')
        plazo_pago_dias = random.choice ([30, 60, 90, 120])
        fecha_vencimiento = fecha_emision + timedelta (days=plazo_pago_dias)
        monto_original = round (random.uniform (10000, 250000), 2)
        estado = random.choices (estados_cuenta, weights=[0.6, 0.3, 0.1])[0]

        if estado == 'Pagada':
            monto_cobrado = monto_original
        elif estado == 'Vencida':
            monto_cobrado = round (monto_original * random.uniform (0, 0.5), 2)
        else:  # Vigente
            monto_cobrado = round (monto_original * random.uniform (0, 0.8), 2)

        cuentas_a_cobrar.append ({
            'factura_id': f'FC-{20000 + i}',
            'cliente_id': cliente['cliente_id'],
            'nombre_cliente': cliente['nombre_cliente'],
            'fecha_emision': fecha_emision,
            'fecha_vencimiento': fecha_vencimiento,
            'plazo_pago_dias': plazo_pago_dias,
            'monto_original': monto_original,
            'monto_cobrado': monto_cobrado,
            'saldo_pendiente': round (monto_original - monto_cobrado, 2),
            'estado_cuenta': estado
        })

    return pd.DataFrame (cuentas_a_cobrar)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica la detección de anomalías y las reglas heurísticas."""
    df['fecha_emision'] = pd.to_datetime (df['fecha_emision'], errors='coerce')
    df['fecha_vencimiento'] = pd.to_datetime (df['fecha_vencimiento'], errors='coerce')
    df['antiguedad_dias'] = (datetime.now () - df['fecha_emision']).dt.days

    features = ['monto_original', 'plazo_pago_dias', 'antiguedad_dias', 'saldo_pendiente']
    X = df[features].copy ().fillna (0)
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X)

    modelo = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = modelo.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    def alerta_reglas(row):
        alertas = []
        hoy = pd.to_datetime ("today")
        if row['estado_cuenta'] == 'Vencida' and row['saldo_pendiente'] > 0:
            dias_vencida = (hoy - row['fecha_vencimiento']).days
            if dias_vencida > 90:
                alertas.append (f"Vencida > 90 días ({dias_vencida} días)")
        if row['saldo_pendiente'] > row['monto_original'] or row['saldo_pendiente'] < 0:
            alertas.append ("Saldo inconsistente")
        return " | ".join (alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply (alerta_reglas, axis=1)
    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("💸 Auditoría de Cuentas a Cobrar")
st.markdown (
    "Esta aplicación realiza una auditoría de cuentas a cobrar simuladas, detectando anomalías con **Isolation Forest** y aplicando reglas heurísticas.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría, por favor espere...'):
        df_cuentas = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_cuentas)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Cuentas", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "Sin alertas")]

        st.subheader ("Cuentas Anómalas o con Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['factura_id', 'nombre_cliente', 'monto_original', 'saldo_pendiente', 'estado_cuenta',
                                'antiguedad_dias', 'resultado_auditoria', 'alerta_heuristica']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_cuentas_a_cobrar.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron cuentas anómalas o con alertas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Distribución de Antigüedad
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        sns.histplot (df_auditado['antiguedad_dias'], bins=15, kde=True, color='skyblue', ax=ax1)
        ax1.set_title ('Distribución de Antigüedad de Cuentas')
        ax1.set_xlabel ('Antigüedad (días)')
        ax1.set_ylabel ('Frecuencia')
        st.pyplot (fig1)

        # Gráfico 2: Anomalias en Saldo vs Antigüedad
        fig2, ax2 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            data=df_auditado, x='antiguedad_dias', y='saldo_pendiente',
            hue='resultado_auditoria', style='estado_cuenta', size='monto_original',
            sizes=(50, 250), palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.7, ax=ax2
        )
        ax2.set_title ('Saldo Pendiente vs. Antigüedad (Detección de Anomalías)')
        ax2.set_xlabel ('Antigüedad (días)')
        ax2.set_ylabel ('Saldo Pendiente ($)')
        ax2.legend (title='Resultado Auditoría')
        st.pyplot (fig2)