# =================================================================
# SCRIPT DE AUDITORÍA DE ACTIVOS INTANGIBLES CON STREAMLIT Y DOCKER
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

# =================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# =================================================================

st.set_page_config (page_title="Auditoría de Activos Intangibles", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de activos intangibles para la auditoría."""
    np.random.seed (789)
    random.seed (789)
    fake = Faker ('es_AR')
    Faker.seed (789)

    num_activos_intangibles = 30
    tipos_activo_intangible = ['Software Licencia', 'Patente', 'Marca Registrada', 'Derechos de Autor',
                               'Fondo de Comercio', 'Lista de Clientes', 'Tecnología no Patentada']
    metodos_amortizacion = ['Línea Recta']
    vida_util_rangos = {'Software Licencia': (3, 7), 'Patente': (10, 20), 'Marca Registrada': (5, 15),
                        'Derechos de Autor': (5, 10), 'Fondo de Comercio': (5, 10), 'Lista de Clientes': (2, 5),
                        'Tecnología no Patentada': (3, 8)}
    empresas_propietarias = [
        {'empresa_id': 3000 + i, 'nombre_empresa': fake.company (), 'cuit': fake.unique.bothify (text='30-########-#')}
        for i in range (10)]

    activos_intangibles = []
    for i in range (num_activos_intangibles):
        propietaria = random.choice (empresas_propietarias)
        tipo = random.choice (tipos_activo_intangible)
        fecha_adquisicion = fake.date_between (start_date='-10y', end_date='-30d')
        min_vida, max_vida = vida_util_rangos.get (tipo, (5, 10))
        vida_util_anios = random.randint (min_vida, max_vida)
        costo_adquisicion = round (random.uniform (50000, 2000000), 2)
        metodo_amortizacion_elegido = random.choice (metodos_amortizacion)
        today = datetime.now ().date ()
        days_since_acquisition = (today - fecha_adquisicion).days
        amortizacion_anual = costo_adquisicion / vida_util_anios
        amortizacion_acumulada_simulada = round (amortizacion_anual * (days_since_acquisition / 365.25), 2)
        amortizacion_acumulada_simulada = min (amortizacion_acumulada_simulada, costo_adquisicion)
        valor_neto_contable_simulado = round (costo_adquisicion - amortizacion_acumulada_simulada, 2)
        if valor_neto_contable_simulado <= 0.01:
            estado = 'Totalmente Amortizado'
        else:
            estado = 'Activo'
        if random.random () < 0.05 and estado == 'Activo':
            estado = 'Vendido'
        activos_intangibles.append ({
            'activo_id': f'INT-{30000 + i}',
            'empresa_id': propietaria['empresa_id'],
            'tipo_activo_intangible': tipo,
            'fecha_adquisicion': fecha_adquisicion,
            'costo_adquisicion': costo_adquisicion,
            'vida_util_anios': vida_util_anios,
            'metodo_amortizacion': metodo_amortizacion_elegido,
            'amortizacion_acumulada_simulada': amortizacion_acumulada_simulada,
            'valor_neto_contable_simulado': valor_neto_contable_simulado,
            'estado_activo': estado,
            'nombre_empresa_propietaria': propietaria['nombre_empresa'],
            'cuit_empresa_propietaria': propietaria['cuit']
        })

    return pd.DataFrame (activos_intangibles)


# =================================================================
# 3. LÓGICA DE AUDITORÍA
# =================================================================

def aplicar_auditoria(df):
    """Aplica las reglas heurísticas y calcula las discrepancias."""
    df['fecha_adquisicion'] = pd.to_datetime (df['fecha_adquisicion'])
    numeric_cols = ['costo_adquisicion', 'vida_util_anios', 'amortizacion_acumulada_simulada',
                    'valor_neto_contable_simulado']
    for col in numeric_cols:
        df[col] = pd.to_numeric (df[col], errors='coerce')
    df.fillna (0, inplace=True)

    df['valor_neto_calculado'] = df['costo_adquisicion'] - df['amortizacion_acumulada_simulada']
    df['discrepancia_vnc'] = df['valor_neto_calculado'] - df['valor_neto_contable_simulado']

    fecha_auditoria = datetime.now ()
    df['amortizacion_anual_calculada'] = df['costo_adquisicion'] / df['vida_util_anios']
    df['amortizacion_anual_calculada'] = df['amortizacion_anual_calculada'].replace ([np.inf, -np.inf], np.nan)
    df['amortizacion_anual_calculada'] = df['amortizacion_anual_calculada'].fillna (0)

    df['anios_transcurridos'] = (fecha_auditoria - df['fecha_adquisicion']).dt.days / 365.25
    df['amortizacion_acumulada_esperada'] = df.apply (
        lambda row: min (row['costo_adquisicion'], row['amortizacion_anual_calculada'] * row['anios_transcurridos']),
        axis=1)
    df['discrepancia_amortizacion'] = df['amortizacion_acumulada_esperada'] - df['amortizacion_acumulada_simulada']
    df['antiguedad_anios'] = (fecha_auditoria - df['fecha_adquisicion']).dt.days / 365.25

    return df


# =================================================================
# 4. INTERFAZ DE STREAMLIT
# =================================================================

st.title ("📊 Auditoría de Activos Intangibles")
st.markdown (
    "Esta aplicación audita datos simulados de activos intangibles, verificando la consistencia de los valores netos y la amortización.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_intangibles = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df_intangibles)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen del Análisis ---
        st.header ("🔍 Informe de Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Activos Analizados", len (df_auditado))
        with col2:
            inconsistencias_vnc = df_auditado[df_auditado['discrepancia_vnc'].abs () > 0.01]
            st.metric ("Activos con Discrepancia en VNC", len (inconsistencias_vnc))

        discrepancias_amort = df_auditado[df_auditado['discrepancia_amortizacion'].abs () > 1]
        if not discrepancias_amort.empty:
            st.subheader ("Activos con Discrepancias en Amortización")
            st.dataframe (discrepancias_amort[['activo_id', 'nombre_empresa_propietaria', 'discrepancia_amortizacion']])
        else:
            st.info ("No se encontraron discrepancias significativas en la amortización.")

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualización de Resultados")

        fig, axes = plt.subplots (2, 3, figsize=(18, 12))
        sns.set_style ("whitegrid")

        # Gráfico 1: Distribución de Tipos de Activos
        df_auditado['tipo_activo_intangible'].value_counts ().plot (kind='bar', color=sns.color_palette ("viridis",
                                                                                                         len (
                                                                                                             df_auditado[
                                                                                                                 'tipo_activo_intangible'].unique ())),
                                                                    ax=axes[0, 0])
        axes[0, 0].set_title ('1. Distribución de Tipos de Activos')
        axes[0, 0].set_ylabel ('Cantidad')
        axes[0, 0].tick_params (axis='x', rotation=45, ha='right')

        # Gráfico 2: Distribución del Costo de Adquisición
        sns.histplot (df_auditado['costo_adquisicion'], bins=10, kde=True, color='skyblue', ax=axes[0, 1])
        axes[0, 1].set_title ('2. Distribución del Costo de Adquisición')

        # Gráfico 3: Costo de Adquisición vs. Vida Útil
        sns.scatterplot (x='vida_util_anios', y='costo_adquisicion', data=df_auditado, hue='tipo_activo_intangible',
                         palette='deep', s=100, ax=axes[0, 2])
        axes[0, 2].set_title ('3. Costo de Adquisición vs. Vida Útil')
        axes[0, 2].legend (title='Tipo', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Gráfico 4: Discrepancias en Amortización
        if not discrepancias_amort.empty:
            discrepancias_amort.set_index ('activo_id')['discrepancia_amortizacion'].plot (kind='bar', color='salmon',
                                                                                           ax=axes[1, 0])
            axes[1, 0].set_title ('4. Discrepancias en Amortización')
            axes[1, 0].set_ylabel ('Monto de Discrepancia')
        else:
            axes[1, 0].text (0.5, 0.5, 'Sin discrepancias.', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title ('4. Discrepancias en Amortización')

        # Gráfico 5: Distribución por Estado del Activo
        df_auditado['estado_activo'].value_counts ().plot (kind='pie', autopct='%1.1f%%', startangle=90,
                                                           colors=sns.color_palette ("pastel"), ax=axes[1, 1])
        axes[1, 1].set_title ('5. Distribución por Estado del Activo')
        axes[1, 1].set_ylabel ('')

        # Gráfico 6: Distribución de Antigüedad
        sns.histplot (df_auditado['antiguedad_anios'], bins=5, kde=True, color='lightgreen', ax=axes[1, 2])
        axes[1, 2].set_title ('6. Distribución de Antigüedad (Años)')

        plt.tight_layout ()
        st.pyplot (fig)