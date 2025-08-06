# =================================================================
# PARTE 0: INSTALACIÓN Y CONFIGURACIÓN
# =================================================================
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st

# =================================================================
# PARTE 1: GENERACIÓN DE DATOS SINTÉTICOS (Función)
# =================================================================
@st.cache_data
def generar_dataframe_previsiones():
    """
    Genera y devuelve un DataFrame de previsiones contables.
    Los datos se crean en memoria.
    """
    st.info("⚙️  Generando conjunto de datos simulado de previsiones...")

    # --- Configuración ---
    np.random.seed(42)
    random.seed(42)
    Faker.seed(42)

    num_previsiones = 30
    fecha_actual_referencia = datetime(2025, 7, 10)

    # Listas de valores posibles
    tipos_prevision = ['Garantías', 'Litigios', 'Cobranzas Dudosas', 'Reestructuración', 'Devoluciones de Ventas',
                       'Desmantelamiento']
    probabilidades = ['Alta', 'Media', 'Baja']
    estados_prevision = ['Activa', 'Utilizada', 'Revertida', 'Ajustada']

    data = []
    for i in range(num_previsiones):
        tipo = random.choice(tipos_prevision)
        estado = random.choices(estados_prevision, weights=[0.6, 0.2, 0.1, 0.1], k=1)[0]
        fecha_creacion = fecha_actual_referencia - timedelta(days=random.randint(30, 365 * 3))
        monto_estimado = round(random.uniform(100000.0, 5000000.0), 2)

        fecha_ult_rev = fecha_creacion + timedelta(days=random.randint(15, 365))
        if fecha_ult_rev > fecha_actual_referencia:
            fecha_ult_rev = fecha_actual_referencia

        fecha_est_utilizacion = pd.NaT
        if estado in ['Activa', 'Ajustada']:
            fecha_est_utilizacion = fecha_actual_referencia + timedelta(days=random.randint(30, 365 * 2))
        elif estado in ['Utilizada', 'Revertida']:
            fecha_est_utilizacion = fecha_creacion + timedelta(days=random.randint(30, 500))
            if fecha_est_utilizacion > fecha_actual_referencia:
                fecha_est_utilizacion = fecha_actual_referencia - timedelta(days=random.randint(1, 60))

        data.append({
            'id_prevision': f'PREV-{i:04d}',
            'tipo_prevision': tipo,
            'descripcion_breve': f"Previsión por {tipo} - Evento {i + 1}",
            'fecha_creacion': fecha_creacion,
            'monto_estimado_ars': monto_estimado,
            'probabilidad_ocurrencia': random.choice(probabilidades),
            'estado_actual': estado,
            'fecha_ultima_revision': fecha_ult_rev,
            'fecha_estimada_utilizacion': fecha_est_utilizacion
        })

    df_previsiones = pd.DataFrame(data)

    for col_fecha in ['fecha_creacion', 'fecha_ultima_revision', 'fecha_estimada_utilizacion']:
        df_previsiones[col_fecha] = pd.to_datetime(df_previsiones[col_fecha])

    st.success("✅ Datos generados exitosamente.")
    return df_previsiones

# =================================================================
# PARTE 2: ANÁLISIS Y VISUALIZACIÓN (Función Principal de Streamlit)
# =================================================================
def analizar_y_visualizar(df_previsiones):
    """
    Procesa, analiza y visualiza los datos en la interfaz de Streamlit.
    """
    st.subheader("🔬 Análisis y Reporte de Auditoría")

    # --- 1. Data Preprocessing ---
    prob_map = {'Baja': 0.25, 'Media': 0.50, 'Alta': 0.75}
    df_previsiones['probabilidad_valor'] = df_previsiones['probabilidad_ocurrencia'].map(prob_map).fillna(0.5)

    fecha_actual_referencia = datetime(2025, 7, 10)
    df_previsiones['dias_desde_creacion'] = (fecha_actual_referencia - df_previsiones['fecha_creacion']).dt.days

    # --- 2. Reporte de Auditoría y Análisis ---
    st.header("📄 REPORTE DE AUDITORÍA DE PREVISIONES")
    st.markdown(f"**Fecha del Reporte:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown(f"**Total de previsiones analizadas:** {len(df_previsiones)}")

    st.subheader("📊 Monto Total Estimado (ARS) por Tipo de Previsión")
    monto_por_tipo = df_previsiones.groupby('tipo_prevision')['monto_estimado_ars'].sum().sort_values(ascending=False)
    st.dataframe(monto_por_tipo.apply(lambda x: f"${x:,.2f}").to_frame())

    # --- 3. Detección de Anomalías (Isolation Forest) ---
    st.subheader("🤖 Detección de Anomalías con Isolation Forest")
    features = ['monto_estimado_ars', 'probabilidad_valor', 'dias_desde_creacion']
    df_ia = df_previsiones[features].copy().fillna(0)

    iso_forest = IsolationForest(random_state=42, contamination=0.1)
    df_previsiones['es_anomalia'] = iso_forest.fit_predict(df_ia)

    anomalias_detectadas = df_previsiones[df_previsiones['es_anomalia'] == -1]
    st.warning(f"Se detectaron {len(anomalias_detectadas)} anomalías potenciales.")
    if not anomalias_detectadas.empty:
        st.dataframe(anomalias_detectadas[['id_prevision', 'tipo_prevision', 'monto_estimado_ars', 'estado_actual']])

    # --- 4. Visualizaciones ---
    st.subheader("📈 Gráficos del Análisis")
    sns.set_style("whitegrid")

    # Gráfico 1: Monto Total por Tipo de Previsión
    st.write('### 1. Monto Total Estimado (ARS) por Tipo de Previsión')
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(x=monto_por_tipo.index, y=monto_por_tipo.values, palette='viridis', ax=ax1)
    ax1.set_ylabel('Monto Total Estimado (ARS)', fontsize=12)
    ax1.set_xlabel('Tipo de Previsión', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # Gráfico 2: Conteo de Previsiones por Estado
    st.write('### 2. Distribución de Previsiones por Estado')
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.countplot(x='estado_actual', data=df_previsiones, palette='cividis',
                   order=df_previsiones['estado_actual'].value_counts().index, ax=ax2)
    ax2.set_xlabel('Estado Actual', fontsize=12)
    ax2.set_ylabel('Cantidad de Previsiones', fontsize=12)
    st.pyplot(fig2)

    # Gráfico 3: Detección de Anomalías
    st.write('### 3. Detección de Anomalías: Monto vs. Antigüedad')
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_previsiones, x='monto_estimado_ars', y='dias_desde_creacion',
        hue='es_anomalia', style='es_anomalia', palette={1: 'blue', -1: 'red'},
        markers={1: 'o', -1: 'X'}, s=100, ax=ax3
    )
    ax3.set_xlabel('Monto Estimado (ARS)', fontsize=12)
    ax3.set_ylabel('Días desde la Creación', fontsize=12)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, ['Normal', 'Anomalía'], title='¿Es Anomalía?')
    st.pyplot(fig3)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Análisis de Previsiones")
    st.title('📈 Análisis de Previsiones Contables')
    st.markdown("""
        Esta aplicación genera datos simulados de previsiones y realiza un análisis
        de auditoría completo, incluyendo la detección de anomalías.
    """)
    df_previsiones_generado = generar_dataframe_previsiones()
    analizar_y_visualizar(df_previsiones_generado)