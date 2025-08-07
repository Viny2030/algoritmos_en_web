# ===============================================================
# SCRIPT DE AUDITORÍA DE MATERIAS PRIMAS CON STREAMLIT Y DOCKER
# ===============================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# ===============================================================

st.set_page_config (page_title="Auditoría de Materias Primas", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de inventario de materias primas."""
    np.random.seed (42)
    random.seed (42)
    fake = Faker ('es_AR')
    Faker.seed (42)

    num_items = 50
    categorias = ['Metales', 'Plásticos', 'Químicos', 'Textiles', 'Madera', 'Papel', 'Vidrio']
    unidades = ['kg', 'litros', 'm3', 'unidades', 'rollos', 'metros']
    descripciones_comunes = ["Cable de Cobre", "Resina Epoxi", "Cloruro de Calcio", "Tela de Algodón",
                             "Madera de Cedro"]

    inventario_data = []
    for i in range (num_items):
        cantidad_stock = round (random.uniform (10, 1000), 2)
        punto_reposicion = round (random.uniform (5, 150), 2)
        costo_unitario = round (random.uniform (1, 500), 2)
        fecha_base = datetime.now () - timedelta (days=random.randint (30, 365))
        inventario_data.append ({
            'id_material': f'MP-{1000 + i}',
            'descripcion': random.choice (descripciones_comunes),
            'categoria': random.choice (categorias),
            'unidad_medida': random.choice (unidades),
            'cantidad_stock': cantidad_stock,
            'punto_reposicion': punto_reposicion,
            'costo_unitario': costo_unitario,
            'valor_total': round (cantidad_stock * costo_unitario, 2),
            'fecha_ultima_entrada': fecha_base.strftime ('%Y-%m-%d'),
            'fecha_ultima_salida': (fecha_base + timedelta (days=random.randint (1, 60))).strftime (
                '%Y-%m-%d') if random.random () > 0.1 else None
        })

    return pd.DataFrame (inventario_data)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_ultima_entrada'] = pd.to_datetime (df['fecha_ultima_entrada'], errors='coerce')
    df['fecha_ultima_salida'] = pd.to_datetime (df['fecha_ultima_salida'], errors='coerce')

    def reglas_auditoria(row):
        alertas = []
        if row['cantidad_stock'] < row['punto_reposicion']:
            alertas.append ("Stock bajo punto de reposición")
        if row['cantidad_stock'] == 0:
            alertas.append ("Sin stock")
        if row['costo_unitario'] <= 0:
            alertas.append ("Costo inválido")
        if pd.isnull (row['fecha_ultima_salida']):
            alertas.append ("Sin uso registrado")
        if pd.notnull (row['fecha_ultima_salida']) and row['fecha_ultima_salida'] < row['fecha_ultima_entrada']:
            alertas.append ("Fecha de salida anterior a entrada")
        return " | ".join (alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply (reglas_auditoria, axis=1)

    features = ['cantidad_stock', 'punto_reposicion', 'costo_unitario', 'valor_total']
    X = df[features].fillna (0)
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X)
    modelo = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = modelo.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("🏭 Auditoría de Materias Primas")
st.markdown (
    "Esta aplicación realiza una auditoría de inventario de materias primas simuladas, identificando anomalías con **Isolation Forest** y aplicando reglas heurísticas de negocio.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_inventario = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_inventario)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Items", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "Sin alertas")]

        st.subheader ("Items Anómalos o con Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_material', 'descripcion', 'categoria', 'cantidad_stock', 'punto_reposicion',
                                'costo_unitario', 'valor_total', 'alerta_heuristica', 'resultado_auditoria']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_inventario.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías o alertas significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Stock vs Punto de Reposición
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        sns.scatterplot (data=df_auditado, x='cantidad_stock', y='punto_reposicion', hue='resultado_auditoria',
                         palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.8, ax=ax1)
        ax1.set_title ('Stock vs Punto de Reposición')
        st.pyplot (fig1)

        # Gráfico 2: Costo Unitario por Categoría
        fig2, ax2 = plt.subplots (figsize=(12, 7))
        sns.boxplot (data=df_auditado, x='costo_unitario', y='categoria', palette='viridis', ax=ax2)
        ax2.set_title ('Distribución del Costo Unitario por Categoría')
        st.pyplot (fig2)

        # Gráfico 3: Valor Total del Inventario vs Cantidad
        fig3, ax3 = plt.subplots (figsize=(10, 6))
        sns.scatterplot (data=df_auditado, x='cantidad_stock', y='valor_total', hue='resultado_auditoria',
                         palette={'Normal': 'blue', 'Anómalo': 'orange'}, alpha=0.8, ax=ax3)
        ax3.set_title ('Valor Total del Inventario vs Cantidad en Stock')
        st.pyplot (fig3)