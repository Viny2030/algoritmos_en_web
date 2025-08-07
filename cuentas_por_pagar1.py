# =================================================================
# SCRIPT DE AUDITORÍA DE CUENTAS POR PAGAR CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st

# =================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# =================================================================

st.set_page_config (page_title="Auditoría de Cuentas por Pagar", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de cuentas por pagar para la auditoría."""
    np.random.seed (42)
    num_registros = 50
    fecha_actual_referencia = datetime.now ()

    ids_factura = [f'INV-{i:04d}' for i in range (num_registros)]
    proveedores = [f'Proveedor_{i}' for i in random.choices (range (1, 21), k=num_registros)]
    montos = [round (random.uniform (100, 75000), 2) for _ in range (num_registros)]
    monedas = random.choices (['USD', 'ARS', 'EUR'], weights=[0.5, 0.4, 0.1], k=num_registros)
    estados_posibles = ['Pendiente', 'Pagada', 'Vencida']

    fechas_emision = []
    fechas_vencimiento = []
    estados_final = []

    for _ in range (num_registros):
        emision = fecha_actual_referencia - timedelta (days=random.randint (10, 365 * 2))
        vencimiento_base = emision + timedelta (days=random.randint (5, 120))
        estado_simulado = random.choices (estados_posibles, weights=[0.65, 0.25, 0.10], k=1)[0]

        if estado_simulado == 'Vencida':
            vencimiento = fecha_actual_referencia - timedelta (days=random.randint (1, 180))
            if emision >= vencimiento:
                emision = vencimiento - timedelta (days=random.randint (5, 60))
        elif estado_simulado == 'Pendiente':
            vencimiento = fecha_actual_referencia + timedelta (days=random.randint (1, 90))
            if emision >= vencimiento:
                emision = vencimiento - timedelta (days=random.randint (5, 60))
        else:  # Pagada
            vencimiento = vencimiento_base

        fechas_emision.append (emision)
        fechas_vencimiento.append (vencimiento)
        estados_final.append (estado_simulado)

    df = pd.DataFrame ({
        'id_factura': ids_factura,
        'proveedor': proveedores,
        'fecha_emision': fechas_emision,
        'fecha_vencimiento': fechas_vencimiento,
        'monto': montos,
        'moneda': monedas,
        'estado': estados_final
    })

    return df


# =================================================================
# 3. LÓGICA DE AUDITORÍA
# =================================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_emision'] = pd.to_datetime (df['fecha_emision'])
    df['fecha_vencimiento'] = pd.to_datetime (df['fecha_vencimiento'])
    fecha_actual_referencia = datetime.now ()

    df['dias_hasta_vencimiento'] = (df['fecha_vencimiento'] - fecha_actual_referencia).dt.days

    bins = [-np.inf, -60, -31, -1, 0, 30, 90, np.inf]
    labels = ['Vencida > 60 días', 'Vencida 31-60 días', 'Vencida 1-30 días',
              'Vence Hoy', 'Por Vencer 1-30 días', 'Por Vencer 31-90 días', 'Por Vencer > 90 días']
    df['rango_antiguedad'] = pd.cut (df['dias_hasta_vencimiento'], bins=bins, labels=labels, right=True)

    df['monto_zscore'] = zscore (df['monto'])
    umbral_zscore = 2.5

    features_for_anomaly_detection = df[['monto', 'dias_hasta_vencimiento']].copy ()
    features_for_anomaly_detection.fillna (features_for_anomaly_detection.median (), inplace=True)
    iso_forest = IsolationForest (random_state=42, contamination=0.1)
    iso_forest.fit (features_for_anomaly_detection)
    df['is_anomaly_ia'] = iso_forest.predict (features_for_anomaly_detection)

    return df


# =================================================================
# 4. INTERFAZ DE STREAMLIT
# =================================================================

st.title ("💸 Auditoría de Cuentas por Pagar")
st.markdown (
    "Esta aplicación audita datos simulados de cuentas por pagar, identificando anomalías en los montos y en los plazos de vencimiento.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_cuentas_pagar = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_cuentas_pagar)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados Clave")

        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total de Facturas", len (df_auditado))
        with col2:
            monto_total_pendiente = df_auditado[df_auditado['estado'] == 'Pendiente']['monto'].sum ()
            st.metric ("Monto Total Pendiente", f"${monto_total_pendiente:,.2f}")
        with col3:
            facturas_vencidas = len (df_auditado[df_auditado['estado'] == 'Vencida'])
            st.metric ("Facturas Vencidas", facturas_vencidas)

        anomalias_zscore = df_auditado[abs (df_auditado['monto_zscore']) > 2.5]
        anomalias_ia = df_auditado[df_auditado['is_anomaly_ia'] == -1]

        st.subheader ("Anomalías Detectadas")
        if not anomalias_ia.empty:
            st.info (
                f"Se encontraron **{len (anomalias_ia)}** facturas con anomalías según el modelo de Isolation Forest.")
            st.dataframe (anomalias_ia[['id_factura', 'proveedor', 'monto', 'dias_hasta_vencimiento', 'is_anomaly_ia']])
        else:
            st.info ("No se encontraron anomalías significativas con el modelo.")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones")

        col_viz1, col_viz2 = st.columns (2)
        with col_viz1:
            # Gráfico 1: Top 10 Proveedores por Monto Total
            monto_por_proveedor = df_auditado.groupby ('proveedor')['monto'].sum ().sort_values (ascending=False)
            fig1, ax1 = plt.subplots (figsize=(12, 7))
            sns.barplot (x=monto_por_proveedor.head (10).index, y=monto_por_proveedor.head (10).values, palette='crest',
                         ax=ax1)
            ax1.set_title ('Top 10 Proveedores por Monto Total a Pagar')
            ax1.set_xlabel ('Proveedor')
            ax1.set_ylabel ('Monto Total a Pagar')
            plt.xticks (rotation=45, ha='right')
            st.pyplot (fig1)

        with col_viz2:
            # Gráfico 2: Conteo de Facturas por Estado
            fig2, ax2 = plt.subplots (figsize=(8, 6))
            sns.countplot (x='estado', data=df_auditado, palette='viridis',
                           order=df_auditado['estado'].value_counts ().index, ax=ax2)
            ax2.set_title ('Distribución de Facturas por Estado')
            ax2.set_xlabel ('Estado de la Factura')
            ax2.set_ylabel ('Número de Facturas')
            st.pyplot (fig2)

        col_viz3, col_viz4 = st.columns (2)
        with col_viz3:
            # Gráfico 3: Monto Total por Estado
            monto_total_por_estado = df_auditado.groupby ('estado')['monto'].sum ().sort_values (ascending=False)
            fig3, ax3 = plt.subplots (figsize=(8, 6))
            sns.barplot (x=monto_total_por_estado.index, y=monto_total_por_estado.values, palette='magma', ax=ax3)
            ax3.set_title ('Monto Total por Estado de Factura')
            ax3.set_xlabel ('Estado de la Factura')
            ax3.set_ylabel ('Monto Total')
            st.pyplot (fig3)

        with col_viz4:
            # Gráfico 4: Monto Total por Rango de Antigüedad
            orden_antiguedad = ['Vencida > 60 días', 'Vencida 31-60 días', 'Vencida 1-30 días', 'Vence Hoy',
                                'Por Vencer 1-30 días', 'Por Vencer 31-90 días', 'Por Vencer > 90 días']
            monto_por_rango_antiguedad = df_auditado.groupby ('rango_antiguedad')['monto'].sum ().reindex (
                orden_antiguedad, fill_value=0)
            fig4, ax4 = plt.subplots (figsize=(14, 8))
            sns.barplot (x=monto_por_rango_antiguedad.index, y=monto_por_rango_antiguedad.values, palette='rocket',
                         ax=ax4)
            ax4.set_title ('Monto Total por Rango de Antigüedad de Deuda')
            ax4.set_xlabel ('Rango de Antigüedad')
            ax4.set_ylabel ('Monto Total')
            plt.xticks (rotation=45, ha='right')
            st.pyplot (fig4)

        # Gráfico 5: Anomalías por IA
        st.subheader ("Anomalías detectadas por Isolation Forest")
        fig5, ax5 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            x='monto',
            y='dias_hasta_vencimiento',
            hue='is_anomaly_ia',
            data=df_auditado,
            palette={1: 'blue', -1: 'red'},
            style='is_anomaly_ia',
            markers={1: 'o', -1: 'X'},
            s=100,
            ax=ax5
        )
        ax5.set_title ('Monto vs. Días hasta Vencimiento')
        ax5.set_xlabel ('Monto de la Factura')
        ax5.set_ylabel ('Días hasta Vencimiento')
        handles, labels = ax5.get_legend_handles_labels ()
        ax5.legend (handles, ['Normal', 'Anomalía'], title='Resultado IA')
        st.pyplot (fig5)