# =================================================================
# SCRIPT DE AUDITORÍA DE PRÉSTAMOS OBTENIDOS CON STREAMLIT Y DOCKER
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

st.set_page_config (page_title="Auditoría de Préstamos Obtenidos", layout="wide")


@st.cache_data
def generar_dataset_prestamos_obtenidos(num_prestamos=50, fecha_inicio='2022-01-01',
                                        rango_monto=(10000, 500000),
                                        rango_tasa_interes=(0.05, 0.20),
                                        rango_plazo_meses=(12, 60),
                                        semilla=42):
    """Genera un DataFrame con datos simulados de préstamos obtenidos."""
    if semilla is not None:
        np.random.seed (semilla)
        random.seed (semilla)

    data = []
    fecha_fin = datetime.now ()
    estados_pago = ['Activo', 'Pagado', 'Atrasado', 'Cancelado']

    for i in range (num_prestamos):
        id_prestamo = f'LOAN-{i + 1:04d}'
        dias_desde_inicio = (fecha_fin - datetime.strptime (fecha_inicio, '%Y-%m-%d')).days
        fecha_obtencion = datetime.strptime (fecha_inicio, '%Y-%m-%d') + timedelta (
            days=np.random.randint (0, dias_desde_inicio))
        monto_prestamo = round (np.random.uniform (rango_monto[0], rango_monto[1]), 2)
        tasa_interes_anual = round (np.random.uniform (rango_tasa_interes[0], rango_tasa_interes[1]), 4)
        plazo_meses = np.random.randint (rango_plazo_meses[0], rango_plazo_meses[1] + 1)
        estado = np.random.choice (estados_pago, p=[0.6, 0.2, 0.15, 0.05])

        data.append ({
            'ID_Prestamo': id_prestamo,
            'Fecha_Obtencion': fecha_obtencion.strftime ('%Y-%m-%d'),
            'Monto_Prestamo': monto_prestamo,
            'Tasa_Interes_Anual': tasa_interes_anual,
            'Plazo_Meses': plazo_meses,
            'Estado_Pago': estado
        })

    df = pd.DataFrame (data)
    df = df.sort_values (by='Fecha_Obtencion').reset_index (drop=True)
    return df


# =================================================================
# 3. LÓGICA DE AUDITORÍA
# =================================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['Fecha_Obtencion'] = pd.to_datetime (df['Fecha_Obtencion'])
    numeric_cols = ['Monto_Prestamo', 'Tasa_Interes_Anual', 'Plazo_Meses']
    for col in numeric_cols:
        df[col] = pd.to_numeric (df[col], errors='coerce')
    df.fillna (0, inplace=True)

    df['Pago_Total_Proyectado'] = df['Monto_Prestamo'] * (1 + df['Tasa_Interes_Anual'] * (df['Plazo_Meses'] / 12))
    fecha_actual_referencia = datetime.now ()
    df['Dias_Desde_Obtencion'] = (fecha_actual_referencia - df['Fecha_Obtencion']).dt.days

    df['monto_zscore'] = zscore (df['Monto_Prestamo'])
    df['tasa_zscore'] = zscore (df['Tasa_Interes_Anual'])

    features_for_anomaly_detection = df[['Monto_Prestamo', 'Tasa_Interes_Anual', 'Plazo_Meses']].copy ()
    features_for_anomaly_detection.fillna (features_for_anomaly_detection.median (), inplace=True)
    iso_forest = IsolationForest (random_state=42, contamination=0.1)
    iso_forest.fit (features_for_anomaly_detection)
    df['is_anomaly_ia'] = iso_forest.predict (features_for_anomaly_detection)

    return df


# =================================================================
# 4. INTERFAZ DE STREAMLIT
# =================================================================

st.title ("💸 Auditoría de Préstamos Obtenidos")
st.markdown (
    "Esta aplicación audita datos simulados de préstamos, identificando anomalías en montos y tasas de interés.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_prestamos = generar_dataset_prestamos_obtenidos (num_prestamos=50, semilla=42)
        df_auditado = aplicar_auditoria (df_prestamos)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Informe de Auditoría")

        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total de Préstamos", len (df_auditado))
        with col2:
            monto_total = df_auditado['Monto_Prestamo'].sum ()
            st.metric ("Monto Total de Préstamos", f"${monto_total:,.2f}")
        with col3:
            anomalias_ia_count = (df_auditado['is_anomaly_ia'] == -1).sum ()
            st.metric ("Anomalías por IA", anomalias_ia_count)

        st.subheader ("Resumen de Estados de Pago")
        st.dataframe (df_auditado['Estado_Pago'].value_counts ())

        anomalias_ia_df = df_auditado[df_auditado['is_anomaly_ia'] == -1]

        if not anomalias_ia_df.empty:
            st.subheader ("Préstamos con Anomalías Detectadas")
            st.dataframe (anomalias_ia_df[['ID_Prestamo', 'Monto_Prestamo', 'Tasa_Interes_Anual', 'Plazo_Meses']])
            csv_data = anomalias_ia_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_prestamos.csv",
                mime="text/csv"
            )
        else:
            st.info ("No se detectaron anomalías por Isolation Forest.")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones")

        # Gráficos de distribución
        col_viz1, col_viz2, col_viz3 = st.columns (3)
        with col_viz1:
            fig1, ax1 = plt.subplots ()
            sns.histplot (df_auditado['Monto_Prestamo'], bins=10, kde=True, color='skyblue', ax=ax1)
            ax1.set_title ('1. Distribución del Monto de Préstamo')
            st.pyplot (fig1)

        with col_viz2:
            fig2, ax2 = plt.subplots ()
            sns.histplot (df_auditado['Tasa_Interes_Anual'], bins=10, kde=True, color='lightgreen', ax=ax2)
            ax2.set_title ('2. Distribución de la Tasa de Interés')
            st.pyplot (fig2)

        with col_viz3:
            fig3, ax3 = plt.subplots ()
            sns.histplot (df_auditado['Plazo_Meses'], bins=5, kde=True, color='salmon', ax=ax3)
            ax3.set_title ('3. Distribución del Plazo (Meses)')
            st.pyplot (fig3)

        # Gráfico 4: Distribución por Estado
        fig4, ax4 = plt.subplots ()
        df_auditado['Estado_Pago'].value_counts ().plot (kind='pie', autopct='%1.1f%%', startangle=90,
                                                         colors=sns.color_palette ("pastel"), ax=ax4)
        ax4.set_title ('4. Distribución de Préstamos por Estado de Pago')
        ax4.set_ylabel ('')
        st.pyplot (fig4)

        # Gráfico 5: Detección de Anomalías con IA
        st.subheader ("Detección de Anomalías por Isolation Forest")
        fig5, ax5 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            x='Monto_Prestamo',
            y='Tasa_Interes_Anual',
            hue='is_anomaly_ia',
            data=df_auditado,
            palette={1: 'blue', -1: 'red'},
            style='is_anomaly_ia',
            markers={1: 'o', -1: 'X'},
            s=100,
            ax=ax5
        )
        ax5.set_title ('Monto vs. Tasa de Interés')
        ax5.set_xlabel ('Monto del Préstamo')
        ax5.set_ylabel ('Tasa de Interés Anual')
        handles, labels = ax5.get_legend_handles_labels ()
        ax5.legend (handles, ['Normal', 'Anomalía'], title='Resultado IA')
        st.pyplot (fig5)