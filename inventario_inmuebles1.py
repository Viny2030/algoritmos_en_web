# =================================================================
# SCRIPT DE AUDITORÍA DE INVENTARIO DE INMUEBLES CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from pandas.tseries.offsets import DateOffset
import streamlit as st

# =================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# =================================================================

st.set_page_config (page_title="Auditoría de Inventario de Inmuebles", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de inventario de inmuebles para la auditoría."""
    fake = Faker ('es_AR')
    random.seed (101)
    np.random.seed (101)
    Faker.seed (101)

    num_inmuebles = 30
    tipos_inmuebles = [
        "Oficina", "Depósito", "Terreno", "Planta industrial",
        "Local comercial", "Edificio", "Galpón", "Centro logístico"
    ]
    estados_inmueble = ["Operativo", "Arrendado", "Mantenimiento", "Disponible", "Inactivo"]
    ubicaciones = ["CABA", "Gran Buenos Aires", "Córdoba", "Mendoza", "Rosario", "Neuquén", "Salta"]

    inmuebles = []
    for i in range (num_inmuebles):
        tipo = random.choice (tipos_inmuebles)
        direccion = fake.address ().replace ("\n", ", ")
        ciudad = random.choice (ubicaciones)
        estado = random.choices (estados_inmueble, weights=[0.5, 0.2, 0.1, 0.1, 0.1])[0]
        fecha_adquisicion = fake.date_between (start_date='-25y', end_date='-2y')
        valor_adquisicion = round (random.uniform (100000.0, 15000000.0), 2)
        superficie_m2 = round (random.uniform (100.0, 5000.0), 2)
        id_inmueble = f"INM-{1000 + i}"

        inmuebles.append ({
            "id_inmueble": id_inmueble,
            "tipo_inmueble": tipo,
            "direccion": direccion,
            "ubicacion": ciudad,
            "estado": estado,
            "fecha_adquisicion": fecha_adquisicion,
            "valor_adquisicion": valor_adquisicion,
            "superficie_m2": superficie_m2
        })
    return pd.DataFrame (inmuebles)


# =================================================================
# 3. LÓGICA DE AUDITORÍA
# =================================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    if 'fecha_adquisicion' not in df.columns:
        df['fecha_adquisicion'] = pd.to_datetime ('2020-01-01')
    else:
        df['fecha_adquisicion'] = pd.to_datetime (df['fecha_adquisicion'], errors='coerce')
        df['fecha_adquisicion'].fillna (pd.to_datetime ('2020-01-01'), inplace=True)

    if 'fecha_fin_vida_util' not in df.columns:
        df['fecha_fin_vida_util'] = df.apply (
            lambda row: row['fecha_adquisicion'] + DateOffset (years=np.random.randint (50, 100)), axis=1)
    else:
        df['fecha_fin_vida_util'] = pd.to_datetime (df['fecha_fin_vida_util'], errors='coerce')
        df['fecha_fin_vida_util'].fillna (df['fecha_adquisicion'] + DateOffset (years=75), inplace=True)

    fecha_actual_referencia = datetime.now ()
    df['edad_anios'] = ((fecha_actual_referencia - df['fecha_adquisicion']).dt.days / 365.25).round (2)
    df['vida_util_restante_anios'] = ((df['fecha_fin_vida_util'] - fecha_actual_referencia).dt.days / 365.25).round (2)
    df.loc[df['vida_util_restante_anios'] < 0, 'vida_util_restante_anios'] = 0

    df['valor_adquisicion_zscore'] = zscore (df['valor_adquisicion'])
    umbral_zscore = 3
    df['is_anomaly_zscore'] = np.where (
        (df['valor_adquisicion_zscore'] > umbral_zscore) | (df['valor_adquisicion_zscore'] < -umbral_zscore), -1, 1)

    features_for_anomaly_detection = df[
        ['valor_adquisicion', 'edad_anios', 'vida_util_restante_anios', 'superficie_m2']].copy ()
    features_for_anomaly_detection.replace ([np.inf, -np.inf], np.nan, inplace=True)
    features_for_anomaly_detection.fillna (features_for_anomaly_detection.median (), inplace=True)

    iso_forest = IsolationForest (random_state=42, contamination=0.1)
    df['is_anomaly_ia'] = iso_forest.fit_predict (features_for_anomaly_detection)

    df['resultado_auditoria'] = 'Normal'
    df.loc[
        (df['is_anomaly_zscore'] == -1) & (df['is_anomaly_ia'] == -1), 'resultado_auditoria'] = 'Anomalía Z-score e IA'
    df.loc[(df['is_anomaly_zscore'] == -1) & (df['is_anomaly_ia'] != -1), 'resultado_auditoria'] = 'Anomalía Z-score'
    df.loc[(df['is_anomaly_zscore'] != -1) & (df['is_anomaly_ia'] == -1), 'resultado_auditoria'] = 'Anomalía IA'

    return df


# =================================================================
# 4. INTERFAZ DE STREAMLIT
# =================================================================

st.title ("🏛️ Auditoría de Inventario de Inmuebles")
st.markdown (
    "Esta aplicación audita datos simulados de inmuebles, utilizando **Z-score** e **Isolation Forest** para la detección de anomalías.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_inmuebles = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_inmuebles)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Inmuebles", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] != 'Normal'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_df = df_auditado[df_auditado['resultado_auditoria'] != 'Normal']

        st.subheader ("Inmuebles Anómalos")
        if not anomalies_df.empty:
            columnas_interes = ['id_inmueble', 'tipo_inmueble', 'ubicacion', 'valor_adquisicion', 'superficie_m2',
                                'resultado_auditoria']
            st.dataframe (anomalies_df[columnas_interes])

            csv_data = anomalies_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_inmuebles.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Valor Total por Tipo de Inmueble
        valor_total_por_tipo = df_auditado.groupby ('tipo_inmueble')['valor_adquisicion'].sum ().sort_values (
            ascending=False)
        fig1, ax1 = plt.subplots (figsize=(12, 7))
        sns.barplot (x=valor_total_por_tipo.index, y=valor_total_por_tipo.values, palette='viridis', ax=ax1)
        ax1.set_title ('Valor Total de Adquisición por Tipo de Inmueble')
        ax1.set_xlabel ('Tipo de Inmueble')
        ax1.set_ylabel ('Valor Total de Adquisición')
        plt.xticks (rotation=45, ha='right')
        st.pyplot (fig1)

        # Gráfico 2: Conteo de Inmuebles por Ubicación y Estado
        conteo_por_ubicacion_estado = df_auditado.groupby (['ubicacion', 'estado']).size ().unstack (fill_value=0)
        fig2, ax2 = plt.subplots (figsize=(14, 8))
        conteo_por_ubicacion_estado.plot (kind='bar', stacked=True, colormap='Paired', ax=ax2)
        ax2.set_title ('Conteo de Inmuebles por Ubicación y Estado')
        ax2.set_xlabel ('Ubicación')
        ax2.set_ylabel ('Número de Inmuebles')
        plt.xticks (rotation=45, ha='right')
        plt.legend (title='Estado')
        st.pyplot (fig2)

        # Gráfico 3: Distribución de la Edad de los Inmuebles
        fig3, ax3 = plt.subplots (figsize=(10, 6))
        sns.histplot (df_auditado['edad_anios'], bins=10, kde=True, color='skyblue', ax=ax3)
        ax3.set_title ('Distribución de la Antigüedad de los Inmuebles (Años)')
        ax3.set_xlabel ('Antigüedad del Inmueble (Años)')
        ax3.set_ylabel ('Frecuencia')
        st.pyplot (fig3)

        # Gráfico 4: Detección de Anomalías en Valor de Adquisición (Z-score)
        fig4, ax4 = plt.subplots (figsize=(12, 7))
        sns.scatterplot (x=df_auditado.index, y='valor_adquisicion', data=df_auditado, label='Inmuebles', color='blue',
                         alpha=0.6, ax=ax4)
        anomalias_valor_zscore = df_auditado[df_auditado['is_anomaly_zscore'] == -1]
        if not anomalias_valor_zscore.empty:
            sns.scatterplot (x=anomalias_valor_zscore.index, y='valor_adquisicion', data=anomalias_valor_zscore,
                             color='red', s=120, label=f'Anomalía (Z-score > {umbral_zscore})', marker='X', ax=ax4)
        ax4.axhline (df_auditado['valor_adquisicion'].mean (), color='gray', linestyle='--', label='Media del Valor')
        ax4.set_title ('Detección de Anomalías en Valor de Adquisición (Z-score)')
        ax4.set_xlabel ('Índice del Inmueble')
        ax4.set_ylabel ('Valor de Adquisición')
        ax4.legend ()
        st.pyplot (fig4)

        # Gráfico 5: Detección de Anomalías con IA (Isolation Forest)
        fig5, ax5 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            x='valor_adquisicion',
            y='superficie_m2',
            hue='is_anomaly_ia',
            data=df_auditado,
            palette={1: 'blue', -1: 'red'},
            style='is_anomaly_ia',
            markers={1: 'o', -1: 'X'},
            s=100,
            ax=ax5
        )
        ax5.set_title ('Detección de Anomalías (IA): Valor Adquisición vs. Superficie (m²)')
        ax5.set_xlabel ('Valor de Adquisición')
        ax5.set_ylabel ('Superficie (m²)')
        handles, labels = ax5.get_legend_handles_labels ()
        new_labels = ['Normal', 'Anomalía IA']
        ax5.legend (handles, new_labels, title='Resultado IA')
        ax5.grid (True, linestyle='--', alpha=0.7)
        st.pyplot (fig5)