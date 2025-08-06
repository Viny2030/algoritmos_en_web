# =================================================================
# PARTE 0: INSTALACIÓN Y CONFIGURACIÓN
# =================================================================
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
def generar_datos():
    # Inicialización
    np.random.seed (42)
    random.seed (42)
    fake = Faker ('es_AR')
    Faker.seed (42)

    # Parámetros
    num_registros = 50
    fecha_actual_referencia = datetime (2025, 7, 10)
    tipos_impuesto = ['IVA', 'Ganancias', 'Ingresos Brutos', 'Aportes Seguridad Social', 'Bienes Personales']
    estados_pago = ['Pendiente', 'Pagado', 'Vencido']

    data = []
    for i in range (num_registros):
        tipo = random.choice (tipos_impuesto)
        periodo_fiscal_date = fecha_actual_referencia - timedelta (days=random.randint (30, 365 * 3))

        if tipo in ['Ganancias', 'Bienes Personales']:
            periodo_fiscal = f"Año {periodo_fiscal_date.year}"
            fecha_venc = datetime (periodo_fiscal_date.year + 1, 4, 15)
        else:
            periodo_fiscal = f"{periodo_fiscal_date.year}-{periodo_fiscal_date.month:02d}"
            fecha_venc = periodo_fiscal_date.replace (day=20) + timedelta (days=30)

        monto = round (random.uniform (50000, 5000000), 2)
        estado = random.choices (estados_pago, weights=[0.6, 0.3, 0.1], k=1)[0]
        fecha_pago = pd.NaT

        if estado == 'Pagado':
            fecha_pago = fecha_venc - timedelta (days=random.randint (0, 10))
        elif estado == 'Vencido':
            fecha_venc = fecha_actual_referencia - timedelta (days=random.randint (1, 90))
        else:
            if fecha_venc <= fecha_actual_referencia:
                fecha_venc = fecha_actual_referencia + timedelta (days=random.randint (1, 60))

        data.append ({
            'id_impuesto': f'IMP-{i:04d}', 'tipo_impuesto': tipo, 'periodo_fiscal': periodo_fiscal,
            'fecha_vencimiento': fecha_venc, 'monto_ars': monto,
            'estado_pago': estado, 'fecha_pago': fecha_pago
        })

    return pd.DataFrame (data), fecha_actual_referencia, tipos_impuesto, estados_pago


# =================================================================
# PARTE 2: ANÁLISIS DE AUDITORÍA Y DETECCIÓN DE ANOMALÍAS (Función)
# =================================================================
def analizar_datos(df_cargas_fiscales, fecha_actual_referencia):
    # Preprocesamiento de Datos
    df_cargas_fiscales['fecha_vencimiento'] = pd.to_datetime (df_cargas_fiscales['fecha_vencimiento'])
    df_cargas_fiscales['fecha_pago'] = pd.to_datetime (df_cargas_fiscales['fecha_pago'])
    df_cargas_fiscales['monto_ars'] = pd.to_numeric (df_cargas_fiscales['monto_ars'], errors='coerce').fillna (0)

    # Análisis de Antigüedad
    df_cargas_fiscales['dias_hasta_vencimiento'] = np.nan
    pendientes_o_vencidas_mask = df_cargas_fiscales['estado_pago'].isin (['Pendiente', 'Vencido'])
    df_cargas_fiscales.loc[pendientes_o_vencidas_mask, 'dias_hasta_vencimiento'] = \
        (df_cargas_fiscales['fecha_vencimiento'] - fecha_actual_referencia).dt.days

    bins_antiguedad = [-np.inf, -90, -31, -1, 0, 30, 90, np.inf]
    labels_antiguedad = ['Vencido > 90 días', 'Vencido 31-90 días', 'Vencido 1-30 días',
                         'Vence Hoy', 'Por Vencer 1-30 días', 'Por Vencer 31-90 días', 'Por Vencer > 90 días']
    df_cargas_fiscales['rango_antiguedad'] = pd.cut (df_cargas_fiscales['dias_hasta_vencimiento'],
                                                     bins=bins_antiguedad, labels=labels_antiguedad, right=True)
    df_cargas_fiscales['rango_antiguedad'] = df_cargas_fiscales['rango_antiguedad'].cat.add_categories (
        'Pagado').fillna ('Pagado')
    orden_antiguedad_completo = ['Pagado'] + labels_antiguedad

    # Detección de Anomalías (Z-score)
    df_cargas_fiscales['monto_ars_zscore'] = zscore (df_cargas_fiscales['monto_ars'])
    umbral_zscore = 2.5
    anomalias_monto_fiscal = df_cargas_fiscales[abs (df_cargas_fiscales['monto_ars_zscore']) > umbral_zscore]

    # Detección de Anomalías (Isolation Forest)
    df_active_cargas = df_cargas_fiscales[df_cargas_fiscales['estado_pago'] != 'Pagado'].copy ()
    if not df_active_cargas.empty:
        df_active_cargas['dias_hasta_vencimiento'].fillna (0, inplace=True)
        features_ia = df_active_cargas[['monto_ars', 'dias_hasta_vencimiento']]
        iso_forest = IsolationForest (random_state=42, contamination=0.1)
        iso_forest.fit (features_ia)
        df_active_cargas['is_anomaly_ia'] = iso_forest.predict (features_ia)
        df_cargas_fiscales = df_cargas_fiscales.merge (df_active_cargas[['id_impuesto', 'is_anomaly_ia']],
                                                       on='id_impuesto', how='left')
        df_cargas_fiscales['is_anomaly_ia'].fillna (1, inplace=True)
    else:
        df_cargas_fiscales['is_anomaly_ia'] = 1

    return df_cargas_fiscales, anomalias_monto_fiscal, umbral_zscore, orden_antiguedad_completo


# =================================================================
# PARTE 3: VISUALIZACIÓN Y REPORTE EN STREAMLIT
# =================================================================
def main():
    st.set_page_config (layout="wide", page_title="Análisis de Cargas Fiscales")
    st.title ('📊 Auditoría y Detección de Anomalías en Cargas Fiscales')

    st.markdown ("""
        Esta aplicación genera datos sintéticos de obligaciones fiscales y realiza un análisis
        para identificar posibles anomalías y generar un reporte de auditoría.
    """)

    df_cargas_fiscales, fecha_actual_referencia, tipos_impuesto, estados_pago = generar_datos ()
    df_cargas_fiscales, anomalias_monto_fiscal, umbral_zscore, orden_antiguedad_completo = analizar_datos (
        df_cargas_fiscales, fecha_actual_referencia)

    st.subheader ('📝 Resumen del Reporte de Auditoría')

    col1, col2, col3 = st.columns (3)

    total_obligaciones = len (df_cargas_fiscales)
    anomalias_zscore_count = len (anomalias_monto_fiscal)
    anomalias_ia_count = (df_cargas_fiscales['is_anomaly_ia'] == -1).sum ()

    col1.metric ("Total de Obligaciones", total_obligaciones)
    col2.metric ("Anomalías por Z-score", anomalias_zscore_count)
    col3.metric ("Anomalías por Isolation Forest", anomalias_ia_count)

    if anomalias_ia_count > 0:
        st.info (
            f"IDs de obligaciones anómalas por Isolation Forest: {df_cargas_fiscales[df_cargas_fiscales['is_anomaly_ia'] == -1]['id_impuesto'].tolist ()}")

    st.subheader ('🔍 Datos Generados (Vista Previa)')
    st.dataframe (df_cargas_fiscales.head (10))

    st.subheader ('📈 Visualización de Resultados')

    # Gráfico 1: Total Amount by Tax Type
    st.markdown ("---")
    st.write ('### 1. Monto Total por Tipo de Impuesto')
    fig1, ax1 = plt.subplots (figsize=(12, 7))
    monto_por_tipo_impuesto = df_cargas_fiscales.groupby ('tipo_impuesto')['monto_ars'].sum ().sort_values (
        ascending=False)
    sns.barplot (x=monto_por_tipo_impuesto.index, y=monto_por_tipo_impuesto.values, palette='viridis', ax=ax1)
    ax1.set_title ('Monto Total (ARS) por Tipo de Impuesto')
    ax1.tick_params (axis='x', rotation=45)
    st.pyplot (fig1)

    # Gráfico 2 & 3: Count and Amount by Status
    st.markdown ("---")
    st.write ('### 2. Distribución de Obligaciones por Estado')
    fig2, (ax2, ax3) = plt.subplots (1, 2, figsize=(16, 6))
    sns.countplot (x='estado_pago', data=df_cargas_fiscales, palette='cividis',
                   order=df_cargas_fiscales['estado_pago'].value_counts ().index, ax=ax2)
    ax2.set_title ('Distribución por Estado')

    monto_total_por_estado_pago = df_cargas_fiscales.groupby ('estado_pago')['monto_ars'].sum ().sort_values (
        ascending=False)
    sns.barplot (x=monto_total_por_estado_pago.index, y=monto_total_por_estado_pago.values, palette='plasma', ax=ax3)
    ax3.set_title ('Monto Total (ARS) por Estado')

    st.pyplot (fig2)

    # Gráfico 4: Amount by Aging Range
    st.markdown ("---")
    st.write ('### 3. Monto Total por Rango de Antigüedad')
    fig4, ax4 = plt.subplots (figsize=(12, 7))
    monto_por_rango_antiguedad = df_cargas_fiscales.groupby ('rango_antiguedad')['monto_ars'].sum ()
    sns.barplot (x=monto_por_rango_antiguedad.index, y=monto_por_rango_antiguedad.values, palette='rocket',
                 order=orden_antiguedad_completo, ax=ax4)
    ax4.set_title ('Monto Total (ARS) por Rango de Antigüedad')
    ax4.tick_params (axis='x', rotation=45)
    st.pyplot (fig4)

    # Gráfico 5: Distribution of Amounts
    st.markdown ("---")
    st.write ('### 4. Distribución de los Montos de Obligación')
    fig5, ax5 = plt.subplots (figsize=(10, 6))
    sns.histplot (df_cargas_fiscales['monto_ars'], bins=15, kde=True, color='purple', ax=ax5)
    ax5.set_title ('Distribución de los Montos (ARS)')
    st.pyplot (fig5)

    # Gráfico 6: Z-score Anomaly Detection
    st.markdown ("---")
    st.write ('### 5. Detección de Anomalías por Monto (Z-score)')
    fig6, ax6 = plt.subplots (figsize=(12, 7))
    sns.scatterplot (x=df_cargas_fiscales.index, y='monto_ars', data=df_cargas_fiscales, label='Obligaciones',
                     color='blue', alpha=0.6, ax=ax6)
    if not anomalias_monto_fiscal.empty:
        sns.scatterplot (x=anomalias_monto_fiscal.index, y='monto_ars', data=anomalias_monto_fiscal,
                         color='red', s=120, label=f'Anomalía (Z-score > {umbral_zscore})', marker='X', ax=ax6)
    ax6.set_title ('Detección de Anomalías por Monto (Z-score)')
    ax6.legend ()
    st.pyplot (fig6)

    # Gráfico 7: Isolation Forest Anomaly Detection
    st.markdown ("---")
    st.write ('### 6. Detección de Anomalías (IA): Monto vs. Días hasta Vencimiento')
    fig7, ax7 = plt.subplots (figsize=(12, 8))
    sns.scatterplot (x='monto_ars', y='dias_hasta_vencimiento', hue='is_anomaly_ia',
                     data=df_cargas_fiscales[df_cargas_fiscales['estado_pago'] != 'Pagado'],
                     palette={1: 'blue', -1: 'red'}, style='is_anomaly_ia', markers={1: 'o', -1: 'X'}, s=100, ax=ax7)
    ax7.set_title ('Detección de Anomalías (IA) en Obligaciones Activas')
    handles, labels = ax7.get_legend_handles_labels ()
    ax7.legend (handles, ['Normal', 'Anomalía'], title='Resultado IA')
    st.pyplot (fig7)

    # Gráfico 8: Box Plot of Amount by Tax Type
    st.markdown ("---")
    st.write ('### 7. Distribución de Montos por Tipo de Impuesto')
    fig8, ax8 = plt.subplots (figsize=(12, 8))
    sns.boxplot (x='tipo_impuesto', y='monto_ars', data=df_cargas_fiscales, palette='pastel', ax=ax8)
    ax8.set_title ('Distribución de Montos por Tipo de Impuesto')
    ax8.tick_params (axis='x', rotation=45)
    st.pyplot (fig8)


if __name__ == '__main__':
    main ()