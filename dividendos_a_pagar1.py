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

    num_dividendos = 50
    fecha_actual_referencia = datetime (2025, 7, 10)
    tipos_accionista = ['Local', 'Extranjero']
    accionistas_locales = [f'Accionista_L{i}' for i in range (1, 21)]
    accionistas_extranjeros = [f'Accionista_E{i}' for i in range (1, 11)]
    monedas = ['ARS', 'USD']
    estados_pago = ['Pendiente', 'Pagado', 'Retenido']

    data = []
    for i in range (num_dividendos):
        tipo_acc = random.choices (tipos_accionista, weights=[0.7, 0.3], k=1)[0]
        accionista_id = random.choice (accionistas_locales) if tipo_acc == 'Local' else random.choice (
            accionistas_extranjeros)
        moneda_div = 'USD' if tipo_acc == 'Extranjero' or random.random () < 0.2 else 'ARS'
        fecha_decl = fecha_actual_referencia - timedelta (days=random.randint (30, 365))
        fecha_pago_est = fecha_decl + timedelta (days=random.randint (15, 90))
        monto_total_dividendo = round (random.uniform (1000, 100000), 2)
        estado = random.choices (estados_pago, weights=[0.6, 0.3, 0.1], k=1)[0]
        fecha_pago_real = pd.NaT

        if estado == 'Pagado':
            fecha_pago_real = fecha_pago_est - timedelta (days=random.randint (-5, 5))
            if fecha_pago_real > fecha_actual_referencia:
                fecha_pago_real = fecha_actual_referencia - timedelta (days=random.randint (1, 30))
            if fecha_pago_real < fecha_decl:
                fecha_pago_real = fecha_decl + timedelta (days=1)
        elif estado == 'Pendiente' and fecha_pago_est <= fecha_actual_referencia:
            fecha_pago_est = fecha_actual_referencia + timedelta (days=random.randint (5, 60))

        data.append ({
            'id_dividendo': f'DIV-{i:04d}', 'tipo_accionista': tipo_acc, 'accionista': accionista_id,
            'fecha_declaracion': fecha_decl, 'fecha_pago_estimada': fecha_pago_est,
            'monto_por_accion': round (monto_total_dividendo / random.randint (100, 10000), 2),
            'cantidad_acciones': random.randint (100, 10000),
            'monto_total_dividendo': monto_total_dividendo, 'moneda': moneda_div,
            'estado_pago': estado, 'fecha_pago_real': fecha_pago_real
        })

    return pd.DataFrame (data), fecha_actual_referencia


# =================================================================
# PARTE 2: ANÁLISIS DE AUDITORÍA Y DETECCIÓN DE ANOMALÍAS (Función)
# =================================================================
@st.cache_data
def analizar_datos(df_dividendos, fecha_actual_referencia):
    df_dividendos['fecha_declaracion'] = pd.to_datetime (df_dividendos['fecha_declaracion'])
    df_dividendos['fecha_pago_estimada'] = pd.to_datetime (df_dividendos['fecha_pago_estimada'])
    df_dividendos['fecha_pago_real'] = pd.to_datetime (df_dividendos['fecha_pago_real'])

    df_dividendos['dias_hasta_pago_est'] = np.nan
    pendientes_o_retenidos_mask = df_dividendos['estado_pago'].isin (['Pendiente', 'Retenido'])
    df_dividendos.loc[pendientes_o_retenidos_mask, 'dias_hasta_pago_est'] = \
        (df_dividendos['fecha_pago_estimada'] - fecha_actual_referencia).dt.days

    bins_proximidad = [-np.inf, -90, -31, -1, 0, 30, 90, np.inf]
    labels_proximidad = ['Vencido > 90 días', 'Vencido 31-90 días', 'Vencido 1-30 días',
                         'Vence Hoy', 'Por Vencer 1-30 días', 'Por Vencer 31-90 días', 'Por Vencer > 90 días']

    df_dividendos['rango_proximidad'] = pd.cut (df_dividendos['dias_hasta_pago_est'], bins=bins_proximidad,
                                                labels=labels_proximidad, right=True)
    df_dividendos['rango_proximidad'] = df_dividendos['rango_proximidad'].cat.add_categories ('Pagado').fillna (
        'Pagado')
    orden_proximidad_completo = ['Pagado'] + labels_proximidad

    df_dividendos['monto_dividendo_zscore'] = zscore (df_dividendos['monto_total_dividendo'])
    umbral_zscore = 2.5
    anomalias_monto_dividendo = df_dividendos[abs (df_dividendos['monto_dividendo_zscore']) > umbral_zscore]

    return df_dividendos, anomalias_monto_dividendo, umbral_zscore, orden_proximidad_completo


# =================================================================
# PARTE 3: APLICACIÓN STREAMLIT
# =================================================================
def main():
    st.set_page_config (layout="wide", page_title="Análisis de Dividendos")
    st.title ('📊 Auditoría y Análisis de Dividendos a Pagar')
    st.markdown ("Esta aplicación genera datos sintéticos de dividendos y realiza un análisis de auditoría.")

    df_dividendos, fecha_actual_referencia = generar_datos ()
    df_dividendos, anomalias_monto_dividendo, umbral_zscore, orden_proximidad_completo = analizar_datos (df_dividendos,
                                                                                                         fecha_actual_referencia)

    st.subheader ('🔍 Datos Generados (Vista Previa)')
    st.dataframe (df_dividendos.head ())

    st.subheader ('📝 Resumen de Auditoría')

    col1, col2, col3 = st.columns (3)
    col1.metric ("Total de Dividendos", len (df_dividendos))
    col2.metric ("Monto Total Pendiente",
                 f"${df_dividendos[df_dividendos['estado_pago'] == 'Pendiente']['monto_total_dividendo'].sum ():,.2f}")
    col3.metric ("Anomalías por Z-score", len (anomalias_monto_dividendo))

    st.subheader ('📈 Visualización de Resultados')
    sns.set (style="whitegrid")

    # Gráfico 1 & 2: Montos por Tipo de Accionista y Estado
    st.write ('### 1. Montos por Tipo de Accionista y Estado')
    col1, col2 = st.columns (2)
    with col1:
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        monto_por_tipo_accionista = df_dividendos.groupby ('tipo_accionista')[
            'monto_total_dividendo'].sum ().sort_values (ascending=False)
        sns.barplot (x=monto_por_tipo_accionista.index, y=monto_por_tipo_accionista.values, palette='viridis', ax=ax1)
        ax1.set_title ('Monto Total por Tipo de Accionista')
        st.pyplot (fig1)

    with col2:
        fig2, ax2 = plt.subplots (figsize=(10, 6))
        monto_por_estado_pago = df_dividendos.groupby ('estado_pago')['monto_total_dividendo'].sum ().sort_values (
            ascending=False)
        sns.barplot (x=monto_por_estado_pago.index, y=monto_por_estado_pago.values, palette='plasma', ax=ax2)
        ax2.set_title ('Monto Total por Estado de Pago')
        st.pyplot (fig2)

    # Gráfico 3: Conteo por Estado
    st.write ('### 2. Distribución de Dividendos por Estado')
    fig3, ax3 = plt.subplots (figsize=(10, 6))
    sns.countplot (x='estado_pago', data=df_dividendos, palette='cividis',
                   order=df_dividendos['estado_pago'].value_counts ().index, ax=ax3)
    ax3.set_title ('Distribución de Dividendos por Estado')
    ax3.set_ylabel ('Número de Dividendos')
    st.pyplot (fig3)

    # Gráfico 4: Monto por Rango de Proximidad
    st.write ('### 3. Monto Total por Proximidad de Pago')
    fig4, ax4 = plt.subplots (figsize=(12, 7))
    monto_por_rango_proximidad = df_dividendos.groupby ('rango_proximidad', observed=True)[
        'monto_total_dividendo'].sum ().reindex (orden_proximidad_completo, fill_value=0)
    sns.barplot (x=monto_por_rango_proximidad.index, y=monto_por_rango_proximidad.values, palette='rocket', ax=ax4)
    ax4.set_title ('Monto Total por Rango de Proximidad de Pago')
    ax4.tick_params (axis='x', rotation=45)
    st.pyplot (fig4)

    # Gráfico 5: Anomalías por Z-score
    st.write ('### 4. Detección de Anomalías en el Monto del Dividendo')
    fig5, ax5 = plt.subplots (figsize=(12, 7))
    sns.scatterplot (x=df_dividendos.index, y='monto_total_dividendo', data=df_dividendos, label='Dividendos',
                     color='blue', alpha=0.6, ax=ax5)
    if not anomalias_monto_dividendo.empty:
        sns.scatterplot (x=anomalias_monto_dividendo.index, y='monto_total_dividendo', data=anomalias_monto_dividendo,
                         color='red', s=120, label=f'Anomalía (Z-score > {umbral_zscore})', marker='X', ax=ax5)
    ax5.axhline (df_dividendos['monto_total_dividendo'].mean (), color='gray', linestyle='--', label='Media del Monto')
    ax5.set_title ('Detección de Anomalías en el Monto del Dividendo')
    ax5.legend ()
    st.pyplot (fig5)


if __name__ == '__main__':
    main ()