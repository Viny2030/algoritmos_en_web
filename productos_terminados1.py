# =================================================================
# SCRIPT DE AUDITORÍA DE PRODUCTOS TERMINADOS CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# ===============================================================

st.set_page_config (page_title="Auditoría de Productos Terminados", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de productos terminados para la auditoría."""
    np.random.seed (42)
    random.seed (42)
    fake = Faker ('es_AR')
    Faker.seed (42)

    num_registros_wip = 50
    productos_base = ['Silla Estándar', 'Mesa de Comedor', 'Estantería Modular', 'Armario Doble', 'Puerta Interior']
    etapas_produccion = ['Corte', 'Ensamblado', 'Soldadura', 'Pintura', 'Control de Calidad']
    lineas_produccion = ['Línea A', 'Línea B', 'Línea C']

    productos_en_proceso_temp = []
    for i in range (num_registros_wip):
        producto = random.choice (productos_base)
        lote = f'L-{fake.random_int (min=1000, max=9999)}'
        cantidad_total = random.randint (50, 200)
        fecha_inicio = fake.date_between (start_date='-60d', end_date='-7d')
        duracion_estimada = random.randint (3, 20)
        fecha_estim_termino = fecha_inicio + timedelta (days=duracion_estimada)
        avance_porcentaje = random.randint (70, 100)
        cantidad_en_proceso_calculada = int (cantidad_total * ((100 - avance_porcentaje) / 100))
        cantidad_terminada_calculada = cantidad_total - cantidad_en_proceso_calculada
        productos_en_proceso_temp.append ({
            'id_proceso': f'WIP-{1000 + i}', 'producto': producto, 'lote': lote,
            'cantidad_terminada_wip': cantidad_terminada_calculada,
            'fecha_estim_termino_proceso': fecha_estim_termino, 'linea_produccion': random.choice (lineas_produccion),
            'avance_porcentaje_wip': avance_porcentaje
        })
    df_wip_temp = pd.DataFrame (productos_en_proceso_temp)

    productos_terminados_final = []
    for _, row in df_wip_temp.iterrows ():
        if row['cantidad_terminada_wip'] > 0:
            fecha_termino_real = row['fecha_estim_termino_proceso'] + timedelta (days=random.randint (0, 5))
            estado_pt = 'En Stock' if row['avance_porcentaje_wip'] == 100 else 'Parcialmente en Almacén'
            productos_terminados_final.append ({
                'id_producto_terminado': f'PT-{row["id_proceso"].split ("-")[1]}', 'producto': row['producto'],
                'lote_produccion': row['lote'], 'linea_produccion': row['linea_produccion'],
                'cantidad_terminada': row['cantidad_terminada_wip'], 'fecha_termino_produccion': fecha_termino_real,
                'estado_almacen': estado_pt, 'costo_produccion_unitario': round (random.uniform (100, 1500), 2),
                'valor_venta_unitario': round (random.uniform (150, 2500), 2)
            })

    return pd.DataFrame (productos_terminados_final)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y el modelo de detección de anomalías."""
    df['fecha_termino_produccion'] = pd.to_datetime (df['fecha_termino_produccion'], errors='coerce')

    def reglas_auditoria(row):
        alertas = []
        if row['cantidad_terminada'] <= 0: alertas.append ("Cantidad inválida")
        if row['costo_produccion_unitario'] <= 0: alertas.append ("Costo inválido")
        if row['valor_venta_unitario'] <= 0: alertas.append ("Valor de venta inválido")
        if row['costo_produccion_unitario'] > row['valor_venta_unitario']: alertas.append (
            "Costo mayor al valor de venta")
        return " | ".join (alertas) if alertas else "OK"

    df['alerta_heuristica'] = df.apply (reglas_auditoria, axis=1)

    features = ['cantidad_terminada', 'costo_produccion_unitario', 'valor_venta_unitario']
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

st.title ("📦 Auditoría de Productos Terminados")
st.markdown (
    "Esta aplicación audita datos simulados de productos terminados, identificando anomalías y aplicando reglas de negocio.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_productos_terminados = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_productos_terminados)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Productos", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_and_alerts_df = df_auditado[
            (df_auditado['resultado_auditoria'] == 'Anómalo') | (df_auditado['alerta_heuristica'] != "OK")]

        st.subheader ("Productos Anómalos o con Alertas")
        if not anomalies_and_alerts_df.empty:
            columnas_interes = ['id_producto_terminado', 'producto', 'alerta_heuristica', 'resultado_auditoria']
            st.dataframe (anomalies_and_alerts_df[columnas_interes])

            csv_data = anomalies_and_alerts_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias_productos_terminados.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías o alertas significativas!")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Costo de Producción por Línea
        fig1, ax1 = plt.subplots (figsize=(9, 5))
        sns.boxplot (data=df_auditado, x='linea_produccion', y='costo_produccion_unitario', ax=ax1)
        ax1.set_title ('Costo de Producción por Línea de Producción')
        st.pyplot (fig1)

        # Gráfico 2: Valor de Venta vs Costo de Producción
        fig2, ax2 = plt.subplots (figsize=(8, 6))
        sns.scatterplot (data=df_auditado, x='costo_produccion_unitario', y='valor_venta_unitario',
                         hue='resultado_auditoria', style='estado_almacen', ax=ax2)
        ax2.set_title ('Valor de Venta vs Costo de Producción')
        st.pyplot (fig2)

        # Gráfico 3: Distribución de Cantidades
        fig3, ax3 = plt.subplots (figsize=(8, 4))
        sns.histplot (df_auditado['cantidad_terminada'], bins=10, kde=True, color='skyblue', ax=ax3)
        ax3.set_title ('Distribución de Cantidades Terminadas')
        st.pyplot (fig3)

        # Gráfico 4: Top 10 Productos con Mayor Margen Unitario
        df_auditado['margen_unitario'] = df_auditado['valor_venta_unitario'] - df_auditado['costo_produccion_unitario']
        top_margen = df_auditado.sort_values (by='margen_unitario', ascending=False).head (10)
        fig4, ax4 = plt.subplots (figsize=(10, 5))
        sns.barplot (data=top_margen, x='producto', y='margen_unitario', palette='viridis', ax=ax4)
        ax4.set_title ('Top 10 Productos con Mayor Margen Unitario')
        ax4.tick_params (axis='x', rotation=45)  # <-- Línea corregida
        plt.tight_layout ()
        st.pyplot (fig4)