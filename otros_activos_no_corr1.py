# =================================================================
# SCRIPT DE AUDITORÍA DE OTROS ACTIVOS CORRIENTES CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# ===============================================================

st.set_page_config (page_title="Auditoría de Otros Activos Corrientes", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de otros activos corrientes para la auditoría."""
    np.random.seed (901)
    random.seed (901)
    fake = Faker ('es_AR')
    Faker.seed (901)

    num_registros = 50
    tipos_activo_corriente = [
        'Valores a cobrar', 'Documentos por cobrar', 'Efectivo y Equivalentes',
        'Inversiones a Corto Plazo', 'Inventarios', 'Cuentas por Cobrar'
    ]
    monedas = ['ARS', 'USD', 'EUR']
    descripciones = [
        'Pago anticipado por publicidad contratada', 'Documentos por cobrar pendientes de pago',
        'Fondos disponibles en caja y bancos', 'Bonos de corto plazo',
        'Materias primas en almacén', 'Facturas de clientes pendientes'
    ]

    activos_corrientes = []
    for i in range (num_registros):
        activos_corrientes.append ({
            'id_activo_corriente': f'AC-{1000 + i}',
            'tipo_activo': random.choice (tipos_activo_corriente),
            'monto': round (random.uniform (10000, 250000), 2),
            'moneda': random.choice (monedas),
            'fecha_registro': fake.date_between (start_date='-120d', end_date='today'),
            'descripcion': random.choice (descripciones)
        })
    return pd.DataFrame (activos_corrientes)


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica el análisis de auditoría a los datos."""
    df['fecha_registro'] = pd.to_datetime (df['fecha_registro'])
    df['monto'] = pd.to_numeric (df['monto'], errors='coerce')
    df.fillna ({'monto': 0}, inplace=True)

    fecha_auditoria = datetime.now ()
    df['dias_desde_registro'] = (fecha_auditoria - df['fecha_registro']).dt.days

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("💰 Auditoría de Otros Activos Corrientes")
st.markdown (
    "Esta aplicación realiza una auditoría y análisis de datos simulados de otros activos corrientes, enfocándose en su distribución y antigüedad.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_activos = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_activos)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Métricas Clave ---
        st.header ("🔍 Resultados Clave del Análisis")

        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total de Registros", len (df_auditado))
        with col2:
            monto_total = df_auditado['monto'].sum ()
            st.metric ("Monto Total ($)", f"{monto_total:,.2f}")
        with col3:
            registros_antiguos = df_auditado[df_auditado['dias_desde_registro'] > 90]
            st.metric ("Registros > 90 días", len (registros_antiguos))

        st.subheader ("Registros Antiguos (> 90 días)")
        if not registros_antiguos.empty:
            st.dataframe (registros_antiguos)
            csv_data = registros_antiguos.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Registros Antiguos CSV",
                data=csv_data,
                file_name="reporte_registros_antiguos.csv",
                mime="text/csv"
            )
        else:
            st.info ("No se encontraron registros con más de 90 días de antigüedad.")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones")

        fig, axes = plt.subplots (2, 2, figsize=(15, 10))
        plt.style.use ('seaborn-v0_8-deep')

        # Gráfico 1: Distribución de Tipos de Activos Corrientes
        df_auditado['tipo_activo'].value_counts ().plot (kind='bar', color=sns.color_palette ("viridis", len (
            df_auditado['tipo_activo'].unique ())), ax=axes[0, 0])
        axes[0, 0].set_title ('1. Distribución de Tipos de Activos Corrientes')
        axes[0, 0].tick_params (axis='x', rotation=45)

        # Gráfico 2: Distribución de Montos por Moneda
        monto_por_moneda = df_auditado.groupby ('moneda')['monto'].sum ().sort_values (ascending=False)
        monto_por_moneda.plot (kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette ("pastel"),
                               ax=axes[0, 1])
        axes[0, 1].set_title ('2. Distribución de Montos por Moneda')
        axes[0, 1].set_ylabel ('')

        # Gráfico 3: Histograma de Monto
        sns.histplot (df_auditado['monto'], bins=15, kde=True, color='skyblue', ax=axes[1, 0])
        axes[1, 0].set_title ('3. Distribución de Montos')

        # Gráfico 4: Histograma de Antigüedad
        sns.histplot (df_auditado['dias_desde_registro'], bins=10, kde=True, color='lightcoral', ax=axes[1, 1])
        axes[1, 1].set_title ('4. Distribución de Antigüedad de Registros (Días)')

        plt.tight_layout ()
        st.pyplot (fig)