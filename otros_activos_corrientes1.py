# =================================================================
# SCRIPT DE AUDITORÍA DE OTROS ACTIVOS CORRIENTES CON STREAMLIT Y DOCKER
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

st.set_page_config (page_title="Auditoría de Otros Activos Corrientes", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de otros activos corrientes para la auditoría."""
    frases = [
        "Pago anticipado por servicios contratados", "Depósito en garantía para contrato vigente",
        "Valores a cobrar por ventas a crédito", "Documentos por cobrar pendientes de pago",
        "Gastos pagados por anticipado de seguros", "Anticipo a proveedor por compra de insumos"
    ]
    np.random.seed (321)
    random.seed (321)
    fake = Faker ('es_ES')
    Faker.seed (321)

    num_registros = 30
    tipos_activos = ['Anticipo a proveedores', 'Depósitos en garantía', 'Valores a cobrar', 'Documentos por cobrar',
                     'Otros créditos', 'Gastos pagados por anticipado']
    monedas = ['ARS', 'USD', 'EUR']

    activos_corrientes = []
    for i in range (num_registros):
        tipo = random.choice (tipos_activos)
        monto = round (random.uniform (1000, 200000), 2)
        moneda = random.choice (monedas)
        fecha_registro = fake.date_between (start_date='-90d', end_date='today')
        descripcion = random.choice (frases)
        activos_corrientes.append ({
            'id_activo_corriente': f'AC-{1000 + i}',
            'tipo_activo': tipo,
            'monto': monto,
            'moneda': moneda,
            'fecha_registro': fecha_registro,
            'descripcion': descripcion
        })
    return pd.DataFrame (activos_corrientes)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_registro'] = pd.to_datetime (df['fecha_registro'], errors='coerce')

    def reglas_auditoria(row):
        alertas = []
        if row['monto'] <= 0: alertas.append ("Monto no válido")
        if row['moneda'] not in ['ARS', 'USD', 'EUR']: alertas.append ("Moneda desconocida")
        if pd.isnull (row['fecha_registro']): alertas.append ("Fecha inválida")
        return " | ".join (alertas) if alertas else "OK"

    df['alerta_heuristica'] = df.apply (reglas_auditoria, axis=1)

    features = ['monto']
    X = df[features].fillna (0)
    X_scaled = StandardScaler ().fit_transform (X)
    modelo = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = modelo.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("💰 Auditoría de Otros Activos Corrientes")
st.markdown (
    "Esta aplicación audita datos simulados de otros activos corrientes, identificando anomalías y aplicando reglas heurísticas.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_activos = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_activos)

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

        st.subheader ("Activos con Anomalías o Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_activo_corriente', 'tipo_activo', 'monto', 'alerta_heuristica',
                                'resultado_auditoria']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_activos_corrientes.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías o alertas significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Distribución de Montos
        fig1, ax1 = plt.subplots (figsize=(8, 4))
        sns.histplot (df_auditado['monto'], bins=15, kde=True, color='skyblue', ax=ax1)
        ax1.set_title ("Distribución de Montos")
        st.pyplot (fig1)

        # Gráfico 2: Montos por Tipo de Activo
        fig2, ax2 = plt.subplots (figsize=(10, 5))
        sns.boxplot (data=df_auditado, x='tipo_activo', y='monto', hue='resultado_auditoria', ax=ax2)
        ax2.set_title ("Montos por Tipo de Activo")
        ax2.tick_params (axis='x', rotation=45)
        st.pyplot (fig2)

        # Gráfico 3: Montos por Moneda
        fig3, ax3 = plt.subplots (figsize=(6, 4))
        sns.boxplot (data=df_auditado, x='moneda', y='monto', ax=ax3)
        ax3.set_title ("Montos por Moneda")
        st.pyplot (fig3)

        # Gráfico 4: Cantidad de Activos Registrados por Mes
        df_auditado['mes'] = df_auditado['fecha_registro'].dt.to_period ("M").astype (str)
        fig4, ax4 = plt.subplots (figsize=(10, 5))
        sns.countplot (data=df_auditado, x='mes', hue='resultado_auditoria', ax=ax4)
        ax4.set_title ("Cantidad de Activos Registrados por Mes")
        ax4.tick_params (axis='x', rotation=45)
        st.pyplot (fig4)