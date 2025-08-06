# =================================================================
# PARTE 0: INSTALACIÓN Y CONFIGURACIÓN
# =================================================================
# --- Importación de Librerías ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st


# =================================================================
# PARTE 1: GENERACIÓN DE DATOS SINTÉTICOS (Función)
# =================================================================
@st.cache_data
def generar_datos():
    np.random.seed (42)
    random.seed (42)
    fake = Faker ('es_AR')
    Faker.seed (42)

    num_deudas = 50
    fecha_actual_referencia = datetime (2025, 7, 10)
    tipos_deuda = ['Préstamo Bancario', 'Deuda Comercial', 'Arrendamiento Financiero', 'Préstamo de Accionista']
    acreedores = [f'Acreedor_{i}' for i in range (1, 21)]
    monedas = ['ARS', 'USD']
    estados_deuda = ['Activa', 'Cancelada', 'Vencida']

    data = []
    for i in range (num_deudas):
        tipo = random.choice (tipos_deuda)
        fecha_orig = fecha_actual_referencia - timedelta (days=random.randint (30, 365 * 5))
        fecha_venc = fecha_orig + timedelta (days=random.randint (90, 365 * 5))
        monto_orig = round (random.uniform (100000, 10000000), 2)
        estado = random.choices (estados_deuda, weights=[0.7, 0.2, 0.1], k=1)[0]

        if estado == 'Cancelada':
            saldo_pendiente = 0.0
            if fecha_venc > fecha_actual_referencia:
                fecha_venc = fecha_actual_referencia - timedelta (days=random.randint (1, 30))
        elif estado == 'Vencida':
            saldo_pendiente = round (random.uniform (0.1, 0.9) * monto_orig, 2)
            fecha_venc = fecha_actual_referencia - timedelta (days=random.randint (1, 180))
            if fecha_orig >= fecha_venc:
                fecha_orig = fecha_venc - timedelta (days=random.randint (30, 365))
        else:  # Activa
            saldo_pendiente = round (monto_orig * random.uniform (0.1, 1.0), 2)
            if fecha_venc <= fecha_actual_referencia:
                fecha_venc = fecha_actual_referencia + timedelta (days=random.randint (1, 365 * 3))

        data.append ({
            'id_deuda': f'DEU-{i:04d}', 'tipo_deuda': tipo, 'acreedor': random.choice (acreedores),
            'fecha_origen': fecha_orig, 'fecha_vencimiento': fecha_venc,
            'monto_original': monto_orig, 'saldo_pendiente': saldo_pendiente,
            'moneda': random.choice (monedas),
            'tasa_interes_anual': round (random.uniform (0.05, 0.45), 4),
            'estado_deuda': estado
        })

    return pd.DataFrame (data), fecha_actual_referencia


# =================================================================
# PARTE 2: ANÁLISIS DE AUDITORÍA Y DETECCIÓN DE ANOMALÍAS (Función)
# =================================================================
@st.cache_data
def analizar_datos(df_otras_deudas, fecha_actual_referencia):
    df_otras_deudas['fecha_origen'] = pd.to_datetime (df_otras_deudas['fecha_origen'])
    df_otras_deudas['fecha_vencimiento'] = pd.to_datetime (df_otras_deudas['fecha_vencimiento'])
    numeric_cols = ['saldo_pendiente', 'tasa_interes_anual']
    for col in numeric_cols:
        df_otras_deudas[col] = pd.to_numeric (df_otras_deudas[col], errors='coerce').fillna (0)

    df_otras_deudas['dias_hasta_vencimiento'] = np.nan
    activa_o_vencida_mask = df_otras_deudas['estado_deuda'].isin (['Activa', 'Vencida'])
    df_otras_deudas.loc[activa_o_vencida_mask, 'dias_hasta_vencimiento'] = \
        (df_otras_deudas['fecha_vencimiento'] - fecha_actual_referencia).dt.days

    bins = [-np.inf, -180, -90, -31, -1, 0, 30, 90, 180, 365, np.inf]
    labels = ['Vencida > 180 días', 'Vencida 91-180 días', 'Vencida 31-90 días', 'Vencida 1-30 días',
              'Vence Hoy', 'Por Vencer 1-30 días', 'Por Vencer 31-90 días',
              'Por Vencer 91-180 días', 'Por Vencer 181-365 días', 'Por Vencer > 365 días']
    df_otras_deudas['rango_vencimiento'] = pd.cut (df_otras_deudas['dias_hasta_vencimiento'], bins=bins, labels=labels,
                                                   right=True)
    df_otras_deudas['rango_vencimiento'] = df_otras_deudas['rango_vencimiento'].cat.add_categories (
        'Cancelada').fillna ('Cancelada')
    orden_vencimiento_completo = ['Cancelada'] + labels

    df_deudas_activas = df_otras_deudas[df_otras_deudas['saldo_pendiente'] > 0].copy ()
    anomalias_saldo = pd.DataFrame ()
    umbral_zscore = 2.5
    if not df_deudas_activas.empty:
        df_deudas_activas['saldo_zscore'] = zscore (df_deudas_activas['saldo_pendiente'])
        anomalias_saldo = df_deudas_activas[abs (df_deudas_activas['saldo_zscore']) > umbral_zscore]

    df_active_deudas_ia = df_otras_deudas[df_otras_deudas['estado_deuda'].isin (['Activa', 'Vencida'])].copy ()
    if not df_active_deudas_ia.empty:
        features_ia = df_active_deudas_ia[['saldo_pendiente', 'dias_hasta_vencimiento']].fillna (0)
        iso_forest = IsolationForest (random_state=42, contamination=0.1)
        iso_forest.fit (features_ia)
        df_active_deudas_ia['is_anomaly_ia'] = iso_forest.predict (features_ia)
        df_otras_deudas = df_otras_deudas.merge (df_active_deudas_ia[['id_deuda', 'is_anomaly_ia']], on='id_deuda',
                                                 how='left')
        df_otras_deudas['is_anomaly_ia'].fillna (1, inplace=True)
    else:
        df_otras_deudas['is_anomaly_ia'] = 1

    return df_otras_deudas, df_deudas_activas, anomalias_saldo, umbral_zscore, orden_vencimiento_completo


# =================================================================
# PARTE 3: APLICACIÓN STREAMLIT
# =================================================================
def main():
    st.set_page_config (layout="wide", page_title="Análisis de Otras Deudas")
    st.title ('🏦 Auditoría y Análisis de Otras Deudas')
    st.markdown ("Esta aplicación genera datos sintéticos de deudas y realiza un análisis de auditoría.")

    df_otras_deudas, df_deudas_activas, anomalias_saldo, umbral_zscore, orden_vencimiento_completo = generar_datos_y_analizar ()

    st.subheader ('🔍 Datos Generados (Vista Previa)')
    st.dataframe (df_otras_deudas.head ())

    st.subheader ('📝 Resumen de Auditoría')

    col1, col2, col3 = st.columns (3)
    col1.metric ("Total de Deudas", len (df_otras_deudas))
    col2.metric ("Saldo Pendiente Total", f"${df_otras_deudas['saldo_pendiente'].sum ():,.2f}")
    col3.metric ("Anomalías por Z-score", len (anomalias_saldo))

    anomalias_ia_count = (df_otras_deudas['is_anomaly_ia'] == -1).sum ()
    st.info (f"Anomalías detectadas por Isolation Forest (en deudas activas/vencidas): {anomalias_ia_count}")

    st.subheader ('📈 Visualización de Resultados')
    sns.set (style="whitegrid")

    # Graph 1 & 2: Saldos por Tipo y Acreedor
    st.write ('### 1. Saldos Pendientes por Tipo de Deuda y Acreedor')
    col1, col2 = st.columns (2)
    with col1:
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        saldo_por_tipo_deuda = df_otras_deudas.groupby ('tipo_deuda')['saldo_pendiente'].sum ().sort_values (
            ascending=False)
        sns.barplot (x=saldo_por_tipo_deuda.index, y=saldo_por_tipo_deuda.values, palette='viridis', ax=ax1)
        ax1.set_title ('Saldo Pendiente por Tipo de Deuda')
        ax1.tick_params (axis='x', rotation=45)
        st.pyplot (fig1)

    with col2:
        fig2, ax2 = plt.subplots (figsize=(10, 6))
        saldo_por_acreedor = df_otras_deudas.groupby ('acreedor')['saldo_pendiente'].sum ().sort_values (
            ascending=False)
        sns.barplot (x=saldo_por_acreedor.head (10).index, y=saldo_por_acreedor.head (10).values, palette='magma',
                     ax=ax2)
        ax2.set_title ('Top 10 Acreedores por Saldo Pendiente')
        ax2.tick_params (axis='x', rotation=45)
        st.pyplot (fig2)

    # Graph 3 & 4: Conteo por Estado y Saldo por Vencimiento
    st.write ('### 2. Distribución y Vencimiento de Deudas')
    col3, col4 = st.columns (2)
    with col3:
        fig3, ax3 = plt.subplots (figsize=(10, 6))
        sns.countplot (x='estado_deuda', data=df_otras_deudas, palette='cividis',
                       order=df_otras_deudas['estado_deuda'].value_counts ().index, ax=ax3)
        ax3.set_title ('Distribución de Deudas por Estado')
        st.pyplot (fig3)

    with col4:
        fig4, ax4 = plt.subplots (figsize=(10, 6))
        saldo_por_rango_vencimiento = df_otras_deudas.groupby ('rango_vencimiento', observed=True)[
            'saldo_pendiente'].sum ().reindex (orden_vencimiento_completo, fill_value=0)
        sns.barplot (x=saldo_por_rango_vencimiento.index, y=saldo_por_rango_vencimiento.values, palette='rocket',
                     ax=ax4)
        ax4.set_title ('Saldo Pendiente por Rango de Vencimiento')
        ax4.tick_params (axis='x', rotation=45)
        st.pyplot (fig4)

    # Graph 5 & 6: Detección de Anomalías
    st.write ('### 3. Detección de Anomalías')
    col5, col6 = st.columns (2)
    with col5:
        fig5, ax5 = plt.subplots (figsize=(10, 6))
        if not df_deudas_activas.empty:
            sns.scatterplot (x=df_deudas_activas.index, y='saldo_pendiente', data=df_deudas_activas,
                             label='Deudas Activas/Vencidas', color='blue', alpha=0.6, ax=ax5)
            if not anomalias_saldo.empty:
                sns.scatterplot (x=anomalias_saldo.index, y='saldo_pendiente', data=anomalias_saldo, color='red', s=120,
                                 label=f'Anomalía (Z-score > {umbral_zscore})', marker='X', ax=ax5)
            ax5.set_title ('Anomalías en Saldos (Z-score)')
            ax5.legend ()
        st.pyplot (fig5)

    with col6:
        fig6, ax6 = plt.subplots (figsize=(10, 6))
        sns.scatterplot (x='saldo_pendiente', y='dias_hasta_vencimiento', hue='is_anomaly_ia', style='is_anomaly_ia',
                         data=df_otras_deudas.dropna (subset=['dias_hasta_vencimiento']),
                         palette={1: 'blue', -1: 'red'}, markers={1: 'o', -1: 'X'}, s=100, ax=ax6)
        ax6.set_title ('Anomalías (IA): Saldo vs. Vencimiento')
        handles, labels = ax6.get_legend_handles_labels ()
        ax6.legend (handles, ['Normal', 'Anomalía'], title='Resultado IA')
        st.pyplot (fig6)

    # Graph 7: Distribución de Tasas de Interés
    st.write ('### 4. Distribución de Tasa de Interés Anual')
    fig7, ax7 = plt.subplots (figsize=(10, 6))
    sns.histplot (df_otras_deudas['tasa_interes_anual'][df_otras_deudas['tasa_interes_anual'] > 0], bins=10, kde=True,
                  color='orange', ax=ax7)
    ax7.set_title ('Distribución de Tasa de Interés Anual')
    st.pyplot (fig7)


def generar_datos_y_analizar():
    df_otras_deudas, fecha_actual_referencia = generar_datos ()
    return analizar_datos (df_otras_deudas, fecha_actual_referencia)


if __name__ == '__main__':
    main ()