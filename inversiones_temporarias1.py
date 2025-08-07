# ===============================================================
# SCRIPT DE AUDITORÍA DE INVERSIONES CON STREAMLIT Y DOCKER
# ===============================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN Y GENERACIÓN DE DATOS SIMULADOS
# ===============================================================

st.set_page_config (page_title="Auditoría de Inversiones", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos de inversiones temporarias simulados."""
    np.random.seed (456)
    random.seed (456)
    fake = Faker ('es_AR')
    Faker.seed (456)
    num_inversiones = 100
    tipos_inversion = ['Fondo Común de Inversión (FCI)', 'Plazo Fijo Bancario', 'Acciones (Cotización)',
                       'Bonos del Tesoro', 'Cauciones Bursátiles']
    plazos_dias = [30, 60, 90, 180, 365]

    empresas_inversoras = [
        {'empresa_id': 2000 + i, 'nombre_empresa': fake.company (), 'cuit': fake.unique.bothify (text='30-########-#')}
        for i in range (20)]
    inversiones = []

    for i in range (num_inversiones):
        inversora = random.choice (empresas_inversoras)
        tipo = random.choice (tipos_inversion)
        fecha_inicio = fake.date_between (start_date='-2y', end_date='today')

        if tipo in ['Plazo Fijo Bancario', 'Cauciones Bursátiles']:
            dias_plazo = random.choice (plazos_dias)
            fecha_vencimiento = fecha_inicio + timedelta (days=dias_plazo)
        else:
            fecha_vencimiento, dias_plazo = None, None

        monto_inicial = round (random.uniform (100000, 5000000), 2)

        tasa_anual_map = {
            'Plazo Fijo Bancario': random.uniform (0.05, 0.12),
            'Fondo Común de Inversión (FCI)': random.uniform (0.03, 0.15),
            'Bonos del Tesoro': random.uniform (0.04, 0.10),
            'Acciones (Cotización)': random.uniform (-0.05, 0.20),
            'Cauciones Bursátiles': random.uniform (0.06, 0.10)
        }
        tasa_anual = round (tasa_anual_map.get (tipo, 0.0), 4)

        if fecha_vencimiento and fecha_vencimiento < datetime.now ().date ():
            estado = 'Vencida' if random.random () <= 0.2 else 'Liquidada'
            monto_final_simulado = monto_inicial * (1 + tasa_anual * (dias_plazo / 365.0))
        else:
            estado = 'Activa'
            days_passed = max (0, (datetime.now ().date () - fecha_inicio).days)
            monto_final_simulado = monto_inicial * (1 + tasa_anual * (days_passed / 365.0) * random.uniform (0.8, 1.2))

        inversiones.append ({
            'inversion_id': f'INV-{20000 + i}',
            'empresa_id': inversora['empresa_id'],
            'tipo_inversion': tipo,
            'entidad_emisora': fake.company () if random.random () > 0.5 else fake.bank (),
            'fecha_inicio': fecha_inicio,
            'fecha_vencimiento': fecha_vencimiento,
            'dias_plazo': dias_plazo,
            'monto_inicial': monto_inicial,
            'tasa_anual_simulada': tasa_anual,
            'valor_actual_simulado': round (monto_final_simulado, 2),
            'estado_inversion': estado,
            'ganancia_perdida_simulada': round (monto_final_simulado - monto_inicial, 2)
        })

    df_inversiones = pd.DataFrame (inversiones)
    df_empresas_inversoras = pd.DataFrame (empresas_inversoras)
    df = pd.merge (df_inversiones, df_empresas_inversoras, on='empresa_id')
    df.sort_values (by='fecha_inicio', inplace=True)
    return df


# ===============================================================
# 3. LÓGICA DE AUDITORÍA
# ===============================================================

def aplicar_auditoria(df):
    """Aplica la detección de anomalías y las reglas heurísticas."""
    df['fecha_inicio'] = pd.to_datetime (df['fecha_inicio'], errors='coerce')
    df['fecha_vencimiento'] = pd.to_datetime (df['fecha_vencimiento'], errors='coerce')

    # Detección de Anomalías (Isolation Forest)
    features = ['monto_inicial', 'tasa_anual_simulada', 'valor_actual_simulado', 'ganancia_perdida_simulada']
    X = df[features].copy ()
    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X.fillna (0))
    iso_forest = IsolationForest (n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly'] = iso_forest.fit_predict (X_scaled)
    df['resultado_auditoria'] = df['anomaly'].map ({1: 'Normal', -1: 'Anómalo'})

    # Reglas Heurísticas
    def auditoria_heuristica(row):
        alertas = []
        hoy = pd.to_datetime ('today')
        if pd.notnull (row['fecha_vencimiento']) and row['fecha_vencimiento'] < hoy and row[
            'estado_inversion'] != 'Liquidada':
            alertas.append ("Vencida no liquidada")
        if row['ganancia_perdida_simulada'] < 0:
            alertas.append ("Pérdida registrada")
        if row['tasa_anual_simulada'] < 0.02 or row['tasa_anual_simulada'] > 0.25:
            alertas.append ("Tasa fuera de rango")
        return " | ".join (alertas) if alertas else "Sin alertas"

    df['alerta_heuristica'] = df.apply (auditoria_heuristica, axis=1)

    return df


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

st.title ("💰 Auditoría de Inversiones Temporarias")
st.markdown (
    "Esta aplicación realiza una auditoría de inversiones simuladas, detectando anomalías con **Isolation Forest** y aplicando reglas heurísticas para identificar posibles riesgos.")

if st.button ("Ejecutar Auditoría", help="Genera datos simulados y aplica el análisis"):
    with st.spinner ('Ejecutando la auditoría, por favor espere...'):
        df = generar_datos_simulados ()
        df_auditado = aplicar_auditoria (df)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resultados de la Auditoría")

        col1, col2 = st.columns (2)
        with col1:
            st.metric ("Total de Inversiones", len (df_auditado))
        with col2:
            anomalias_count = len (df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo'])
            st.metric ("Anomalías Detectadas", anomalias_count)

        anomalies_df = df_auditado[df_auditado['resultado_auditoria'] == 'Anómalo']

        st.subheader ("Inversiones Anómalas y con Alertas")
        if not anomalies_df.empty:
            columnas_interes = ['inversion_id', 'nombre_empresa', 'tipo_inversion', 'estado_inversion', 'monto_inicial',
                                'ganancia_perdida_simulada', 'tasa_anual_simulada', 'alerta_heuristica',
                                'resultado_auditoria']
            st.dataframe (anomalies_df[columnas_interes])

            csv_data = anomalies_df.to_csv (index=False).encode ('utf-8')
            st.download_button (
                label="Descargar Reporte de Anomalías CSV",
                data=csv_data,
                file_name="reporte_anomalias.csv",
                mime="text/csv"
            )
        else:
            st.info ("¡No se encontraron anomalías significativas según el modelo!")

        # --- Sección 2: Visualizaciones Clave ---
        st.header ("📈 Visualizaciones")

        # Gráfico 1: Distribución de Tasas
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        sns.histplot (df_auditado['tasa_anual_simulada'], kde=True, bins=15, color='skyblue', ax=ax1)
        ax1.set_title ('Distribución de Tasas Anuales Simuladas')
        ax1.set_xlabel ('Tasa Anual')
        ax1.set_ylabel ('Frecuencia')
        st.pyplot (fig1)

        # Gráfico 2: Anomalias en Ganancia vs Monto
        fig2, ax2 = plt.subplots (figsize=(12, 8))
        sns.scatterplot (
            data=df_auditado, x='monto_inicial', y='ganancia_perdida_simulada',
            hue='resultado_auditoria', style='estado_inversion', size='tasa_anual_simulada',
            sizes=(50, 250), palette={'Normal': 'green', 'Anómalo': 'red'}, alpha=0.7, ax=ax2
        )
        ax2.set_title ('Monto Inicial vs Ganancia/Pérdida (Detección de Anomalías)')
        ax2.set_xlabel ('Monto Inicial ($)')
        ax2.set_ylabel ('Ganancia / Pérdida Simulada ($)')
        ax2.get_xaxis ().set_major_formatter (plt.FuncFormatter (lambda x, p: format (int (x), ',')))
        ax2.get_yaxis ().set_major_formatter (plt.FuncFormatter (lambda y, p: format (int (y), ',')))
        ax2.legend (title='Resultado Auditoría')
        st.pyplot (fig2)