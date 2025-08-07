# =================================================================
# SCRIPT DE AUDITORÍA DE INVENTARIO DE MAQUINARIAS CON STREAMLIT Y DOCKER
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
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# ===============================================================

st.set_page_config (page_title="Auditoría de Inventario de Maquinarias", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de inventario de maquinarias para la auditoría."""
    fake = Faker ('es_AR')
    random.seed (42)
    np.random.seed (42)
    Faker.seed (42)
    num_equipos = 30
    tipos_maquinaria = ["Torno CNC", "Fresadora", "Impresora industrial", "Camión", "Grua hidráulica"]
    ubicaciones = ["Planta A", "Planta B", "Depósito Central", "Taller de Mantenimiento"]
    estados = ["Operativo", "En reparación", "Fuera de servicio", "Pendiente de baja"]

    maquinarias = []
    for i in range (num_equipos):
        tipo = random.choice (tipos_maquinaria)
        fecha_adquisicion = fake.date_between (start_date='-10y', end_date='-1y')
        valor_adquisicion = round (random.uniform (200000, 5000000), 2)
        vida_util_anios = random.randint (5, 15)
        maquinarias.append ({
            "id_equipo": f"EQ-{1000 + i}",
            "tipo_equipo": tipo,
            "descripcion": f"{tipo} modelo {fake.bothify (text='???-####')}",
            "ubicacion": random.choice (ubicaciones),
            "estado": random.choices (estados, weights=[0.7, 0.15, 0.1, 0.05])[0],
            "fecha_adquisicion": fecha_adquisicion,
            "valor_adquisicion": valor_adquisicion,
            "vida_util_anios": vida_util_anios,
            "fecha_fin_vida_util": fecha_adquisicion + timedelta (days=vida_util_anios * 365)
        })
    return pd.DataFrame (maquinarias)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_adquisicion'] = pd.to_datetime (df['fecha_adquisicion'])
    df['fecha_fin_vida_util'] = pd.to_datetime (df['fecha_fin_vida_util'])
    fecha_actual = datetime.now ()
    df['edad_anios'] = ((fecha_actual - df['fecha_adquisicion']).dt.days / 365.25).round (2)
    df['vida_util_restante_anios'] = ((df['fecha_fin_vida_util'] - fecha_actual).dt.days / 365.25).round (2)
    df.loc[df['vida_util_restante_anios'] < 0, 'vida_util_restante_anios'] = 0
    df['valor_adquisicion_zscore'] = zscore (df['valor_adquisicion'])

    umbral_z = 2.5
    features = df[['valor_adquisicion', 'edad_anios', 'vida_util_restante_anios']].copy ()
    iso = IsolationForest (random_state=42, contamination=0.1)
    df['is_anomaly_ia'] = iso.fit_predict (features)

    df['alerta_combinada'] = df.apply (lambda row: 'Z-score alto y Anomalía IA' if (
                abs (row['valor_adquisicion_zscore']) > umbral_z and row['is_anomaly_ia'] == -1) else (
        'Z-score alto' if abs (row['valor_adquisicion_zscore']) > umbral_z else (
            'Anomalía IA' if row['is_anomaly_ia'] == -1 else 'Sin alerta')), axis=1)

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("🏭 Auditoría de Inventario de Maquinarias")
st.markdown (
    "Esta aplicación audita el inventario de maquinarias simulado, combinando la detección de anomalías por **Z-score** e **Isolation Forest**.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_maquinarias = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_maquinarias)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resultados y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total de Equipos", len (df_auditado))
        with col2:
            anomalias_ia_count = len (df_auditado[df_auditado['is_anomaly_ia'] == -1])
            st.metric ("Anomalías por IA", anomalias_ia_count)
        with col3:
            alertas_combinadas_count = len (df_auditado[df_auditado['alerta_combinada'] != 'Sin alerta'])
            st.metric ("Alertas Combinadas", alertas_combinadas_count)

        st.subheader ("Equipos con Alertas Detectadas")
        alertas_df = df_auditado[df_auditado['alerta_combinada'] != 'Sin alerta']
        if not alertas_df.empty:
            st.dataframe (alertas_df[['id_equipo', 'tipo_equipo', 'valor_adquisicion', 'alerta_combinada']])
            csv_data = alertas_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Alertas CSV",
                data=csv_data,
                file_name="reporte_alertas_maquinarias.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron alertas o anomalías significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        col_viz1, col_viz2 = st.columns (2)
        with col_viz1:
            # Gráfico 1: Valor Total por Tipo de Equipo
            valor_total_tipo = df_auditado.groupby ('tipo_equipo')['valor_adquisicion'].sum ().sort_values (
                ascending=False)
            fig1, ax1 = plt.subplots (figsize=(10, 6))
            sns.barplot (x=valor_total_tipo.index, y=valor_total_tipo.values, palette='viridis', ax=ax1)
            ax1.set_title ('Valor Total de Adquisición por Tipo de Equipo')
            ax1.set_xlabel ('')
            ax1.set_ylabel ('Valor Total ($)')
            plt.xticks (rotation=45, ha='right')
            st.pyplot (fig1)

        with col_viz2:
            # Gráfico 2: Conteo por Ubicación y Estado
            conteo = df_auditado.groupby (['ubicacion', 'estado']).size ().unstack (fill_value=0)
            fig2, ax2 = plt.subplots (figsize=(10, 7))
            conteo.plot (kind='bar', stacked=True, colormap='Paired', ax=ax2)
            ax2.set_title ('Conteo de Equipos por Ubicación y Estado')
            ax2.set_xlabel ('Ubicación')
            ax2.set_ylabel ('Número de Equipos')
            plt.xticks (rotation=45, ha='right')
            st.pyplot (fig2)

        col_viz3, col_viz4 = st.columns (2)
        with col_viz3:
            # Gráfico 3: Distribución de Antigüedad
            fig3, ax3 = plt.subplots (figsize=(10, 5))
            df_auditado['edad_anios'].hist (bins=10, color='skyblue', ec='black', ax=ax3)
            ax3.set_title ('Distribución de Antigüedad de Equipos')
            ax3.set_xlabel ('Antigüedad (años)')
            ax3.set_ylabel ('Frecuencia')
            st.pyplot (fig3)

        with col_viz4:
            # Gráfico 4: Distribución de Vida Útil Restante
            fig4, ax4 = plt.subplots (figsize=(10, 5))
            sns.histplot (df_auditado['vida_util_restante_anios'], bins=10, kde=True, color='orange', ax=ax4)
            ax4.set_title ('Distribución de Vida Útil Restante')
            ax4.set_xlabel ('Vida Útil Restante (años)')
            ax4.set_ylabel ('Frecuencia')
            st.pyplot (fig4)

        # Gráfico 5: Anomalias 3D
        st.subheader ("Anomalías detectadas por Isolation Forest (3D)")
        fig5 = plt.figure (figsize=(12, 9))
        ax5 = fig5.add_subplot (111, projection='3d')
        normal_data = df_auditado[df_auditado['is_anomaly_ia'] == 1]
        anomaly_data = df_auditado[df_auditado['is_anomaly_ia'] == -1]
        ax5.scatter (normal_data['edad_anios'], normal_data['vida_util_restante_anios'],
                     normal_data['valor_adquisicion'], c='blue', label='Normal', alpha=0.6)
        ax5.scatter (anomaly_data['edad_anios'], anomaly_data['vida_util_restante_anios'],
                     anomaly_data['valor_adquisicion'], c='red', label='Anomalía IA', marker='^', s=100)
        ax5.set_xlabel ('Antigüedad (años)')
        ax5.set_ylabel ('Vida Útil Restante (años)')
        ax5.set_zlabel ('Valor de Adquisición ($)')
        ax5.set_title ('Visualización 3D de Anomalías Detectadas por IA')
        ax5.legend ()
        st.pyplot (fig5)

        # Gráfico 6: Resumen de Alertas Combinadas
        st.subheader ("Resumen de Alertas")
        fig6, ax6 = plt.subplots (figsize=(10, 5))
        sns.countplot (x='alerta_combinada', data=df_auditado, palette='Set2',
                       order=df_auditado['alerta_combinada'].value_counts ().index, ax=ax6)
        ax6.set_title ('Resumen de Alertas Combinadas')
        ax6.set_xlabel ('')
        ax6.set_ylabel ('Número de Equipos')
        plt.xticks (rotation=45, ha='right')
        st.pyplot (fig6)