# =================================================================
# PARTE 0: INSTALACIÓN Y CONFIGURACIÓN
# =================================================================
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
import streamlit as st


# =================================================================
# PARTE 1: GENERACIÓN DE DATOS SINTÉTICOS (Función)
# =================================================================
@st.cache_data
def generate_debt_dataframe():
    st.info ("⚙️  Generando conjunto de datos simulado...")

    np.random.seed (1011)
    random.seed (1011)
    fake = Faker ('es_AR')
    Faker.seed (1011)

    num_deudas = 30
    tipos_deuda_no_corriente = [
        'Préstamo Bancario a Largo Plazo', 'Bonos Emitidos', 'Hipoteca Inmobiliaria',
        'Arrendamiento Financiero (Leasing)', 'Deuda con Partes Relacionadas (Largo Plazo)',
        'Obligaciones Negociables'
    ]
    plazos_anios = [3, 5, 7, 10, 15, 20]

    num_empresas_deudoras = 25
    empresas_deudoras = [{
        'empresa_id': 5000 + i,
        'nombre_empresa': fake.company (),
        'cuit': fake.unique.bothify (text='30-########-#')
    } for i in range (num_empresas_deudoras)]

    deudas_no_corrientes = []
    for i in range (num_deudas):
        deudora = random.choice (empresas_deudoras)
        tipo = random.choice (tipos_deuda_no_corriente)
        fecha_emision = fake.date_between (start_date='-10y', end_date='-90d')
        plazo_anios_elegido = random.choice (plazos_anios)
        fecha_vencimiento = fecha_emision + timedelta (days=plazo_anios_elegido * 365.25)
        monto_original = round (random.uniform (500000, 10000000), 2)

        if tipo == 'Préstamo Bancario a Largo Plazo':
            tasa_interes_anual = round (random.uniform (0.06, 0.15), 4)
        elif tipo in ['Bonos Emitidos', 'Obligaciones Negociables']:
            tasa_interes_anual = round (random.uniform (0.04, 0.12), 4)
        else:
            tasa_interes_anual = round (random.uniform (0.03, 0.10), 4)

        today = datetime.now ().date ()
        days_passed = (today - fecha_emision).days

        if fecha_vencimiento < today:
            estado = random.choices (['Pagada', 'Incumplida', 'Refinanciada'], weights=[0.6, 0.2, 0.2])[0]
            saldo_pendiente_simulado = 0.0 if estado == 'Pagada' else round (monto_original * random.uniform (0.1, 1.0),
                                                                             2)
        else:
            estado = 'Activa'
            total_days = (fecha_vencimiento - fecha_emision).days
            saldo_pendiente_simulado = round (monto_original * (1 - (days_passed / total_days)),
                                              2) if total_days > 0 else monto_original
            if saldo_pendiente_simulado < 0: saldo_pendiente_simulado = 0.0

            if random.random () < 0.02:
                estado = 'Incumplida'

        intereses_acumulados_simulados = round (monto_original * tasa_interes_anual * (days_passed / 365.25), 2)
        if intereses_acumulados_simulados < 0: intereses_acumulados_simulados = 0.0

        deudas_no_corrientes.append ({
            'deuda_id': f'DNC-{50000 + i}', 'empresa_id': deudora['empresa_id'], 'tipo_deuda': tipo,
            'fecha_emision': fecha_emision, 'fecha_vencimiento': fecha_vencimiento, 'plazo_anios': plazo_anios_elegido,
            'monto_original': monto_original, 'tasa_interes_anual': tasa_interes_anual,
            'saldo_pendiente_simulado': saldo_pendiente_simulado,
            'intereses_acumulados_simulados': intereses_acumulados_simulados, 'estado_deuda': estado,
            'nombre_empresa_deudora': deudora['nombre_empresa'], 'cuit_empresa_deudora': deudora['cuit']
        })

    df = pd.DataFrame (deudas_no_corrientes)
    df.sort_values (by='fecha_emision', inplace=True)
    st.success ("✅ Datos generados exitosamente.")
    return df


# =================================================================
# PARTE 2: ANÁLISIS DE AUDITORÍA Y VISUALIZACIÓN (Función)
# =================================================================
def analyze_and_visualize(df):
    st.subheader ("🔬 Análisis y Reporte de Auditoría")

    # --- 1. Data Preprocessing ---
    df['fecha_emision'] = pd.to_datetime (df['fecha_emision'])
    df['fecha_vencimiento'] = pd.to_datetime (df['fecha_vencimiento'])
    numeric_cols = ['plazo_anios', 'monto_original', 'tasa_interes_anual', 'saldo_pendiente_simulado',
                    'intereses_acumulados_simulados']
    for col in numeric_cols:
        df[col] = pd.to_numeric (df[col], errors='coerce')
    df.fillna (0, inplace=True)

    # --- 2. Análisis Descriptivo y Reporte de Auditoría ---
    st.markdown ("---")
    st.header ("📄 REPORTE DE AUDITORÍA DE DEUDAS NO CORRIENTES")
    st.markdown (f"**Fecha del Reporte:** {datetime.now ().strftime ('%Y-%m-%d %H:%M:%S')}")
    st.markdown (f"**Total de deudas analizadas:** {len (df)}")

    # --- Métricas clave ---
    col1, col2 = st.columns (2)
    col1.metric ("Monto original total de deudas", f"${df['monto_original'].sum ():,.2f}")
    col2.metric ("Saldo pendiente total", f"${df['saldo_pendiente_simulado'].sum ():,.2f}")

    # --- 3. Detección de Anomalías con Isolation Forest ---
    st.subheader ("🚨 Detección de Anomalías (Isolation Forest)")
    features = ['saldo_pendiente_simulado', 'tasa_interes_anual', 'plazo_anios']
    df_active = df[df['estado_deuda'].isin (['Activa', 'Incumplida'])].copy ()

    if not df_active.empty:
        iso_forest = IsolationForest (random_state=42, contamination=0.1)
        df_active['is_anomaly'] = iso_forest.fit_predict (df_active[features])
        anomalies_count = (df_active['is_anomaly'] == -1).sum ()
        st.write (f"Anomalías detectadas por IA: **{anomalies_count}**")
        if anomalies_count > 0:
            anomalies_df = df_active[df_active['is_anomaly'] == -1]
            st.warning ("Deudas anómalas recomendadas para revisión:")
            st.dataframe (
                anomalies_df[['deuda_id', 'nombre_empresa_deudora', 'tipo_deuda', 'saldo_pendiente_simulado']])
    else:
        st.info ("No hay deudas activas para analizar anomalías.")

    # --- 4. Visualizaciones ---
    st.subheader ("📊 Visualizaciones")
    sns.set (style="whitegrid", palette="viridis")

    # Gráfico 1: Saldo Pendiente por Tipo de Deuda
    fig1, ax1 = plt.subplots (figsize=(12, 7))
    saldo_por_tipo = df.groupby ('tipo_deuda')['saldo_pendiente_simulado'].sum ().sort_values (ascending=False)
    sns.barplot (x=saldo_por_tipo.index, y=saldo_por_tipo.values, ax=ax1)
    ax1.set_title ('1. Saldo Pendiente Total por Tipo de Deuda', fontsize=16)
    ax1.set_ylabel ('Saldo Pendiente Total', fontsize=12)
    ax1.set_xlabel ('Tipo de Deuda', fontsize=12)
    ax1.tick_params (axis='x', rotation=45)
    st.pyplot (fig1)

    # Gráfico 2: Conteo de Deudas por Estado
    fig2, ax2 = plt.subplots (figsize=(8, 6))
    sns.countplot (x='estado_deuda', data=df, order=df['estado_deuda'].value_counts ().index, ax=ax2)
    ax2.set_title ('2. Distribución de Deudas por Estado', fontsize=16)
    ax2.set_xlabel ('Estado de la Deuda', fontsize=12)
    ax2.set_ylabel ('Cantidad de Deudas', fontsize=12)
    st.pyplot (fig2)

    # Gráfico 3: Detección de Anomalías (IA)
    if not df_active.empty:
        fig3, ax3 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            data=df_active,
            x='saldo_pendiente_simulado',
            y='tasa_interes_anual',
            hue='is_anomaly',
            style='is_anomaly',
            palette={1: 'blue', -1: 'red'},
            markers={1: 'o', -1: 'X'},
            s=100,
            ax=ax3
        )
        ax3.set_title ('3. Detección de Anomalías (IA): Saldo vs. Tasa de Interés', fontsize=16)
        ax3.set_xlabel ('Saldo Pendiente', fontsize=12)
        ax3.set_ylabel ('Tasa de Interés Anual', fontsize=12)
        ax3.legend (title='¿Es Anomalía?', labels=['No', 'Sí'])
        st.pyplot (fig3)


if __name__ == "__main__":
    st.set_page_config (layout="wide", page_title="Análisis de Deudas No Corrientes")
    st.title ('📈 Análisis de Deudas No Corrientes')
    st.markdown ("""
        Esta aplicación genera un conjunto de datos simulado de deudas a largo plazo y realiza un análisis de auditoría,
        incluyendo la detección de anomalías.
    """)

    # Generar y analizar el DataFrame
    deudas_df = generate_debt_dataframe ()
    analyze_and_visualize (deudas_df)