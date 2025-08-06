# -*- coding: utf-8 -*-
# =================================================================
# PARTE 0: INSTALACIÓN Y CONFIGURACIÓN
# =================================================================
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st

# =================================================================
# PARTE 1: GENERACIÓN DE DATOS SINTÉTICOS (Función)
# =================================================================
@st.cache_data
def generar_dataframe_patrimonio_neto(num_periodos=30, fecha_inicio='2023-01-01',
                                      patrimonio_neto_inicial_base=500000,
                                      rango_aportes=(10000, 50000),
                                      rango_dividendos=(5000, 20000),
                                      rango_resultado_ejercicio=(-30000, 80000),
                                      prob_aporte=0.2, prob_dividendo=0.15,
                                      prob_resultado_negativo=0.3,
                                      semilla=42):
    """
    Genera y devuelve un DataFrame de pandas que simula un Estado de Patrimonio Neto.
    """
    np.random.seed(semilla)
    fechas_inicio_periodo = pd.date_range(start=fecha_inicio, periods=num_periodos, freq='YS')
    data = []
    patrimonio_neto_al_inicio = patrimonio_neto_inicial_base

    for fecha_actual in fechas_inicio_periodo:
        aportes_capital = np.random.uniform(*rango_aportes) if np.random.rand() < prob_aporte else 0
        dividendos = np.random.uniform(*rango_dividendos) if np.random.rand() < prob_dividendo else 0
        resultado_ejercicio = np.random.uniform(*rango_resultado_ejercicio)

        if np.random.rand() < prob_resultado_negativo:
            resultado_ejercicio = -abs(resultado_ejercicio)

        if dividendos > patrimonio_neto_al_inicio + aportes_capital:
            dividendos = (patrimonio_neto_al_inicio + aportes_capital) * 0.1

        patrimonio_neto_al_cierre = (patrimonio_neto_al_inicio + aportes_capital
                                     - dividendos + resultado_ejercicio)

        if patrimonio_neto_al_cierre < 50000:
            patrimonio_neto_al_cierre = 50000

        data.append({
            'Fecha_Inicio_Periodo': fecha_actual,
            'Patrimonio_Neto_Inicio': round(patrimonio_neto_al_inicio, 2),
            'Aportes_Capital': round(aportes_capital, 2),
            'Menos_Distribuciones_Dividendos': round(dividendos, 2),
            'Resultado_del_Ejercicio': round(resultado_ejercicio, 2),
            'Otras_Variaciones': 0.0,
            'Patrimonio_Neto_Cierre': round(patrimonio_neto_al_cierre, 2)
        })
        patrimonio_neto_al_inicio = patrimonio_neto_al_cierre

    df = pd.DataFrame(data)
    st.success("✅ Datos de Patrimonio Neto generados exitosamente.")
    return df

# =================================================================
# PARTE 2: ANÁLISIS Y VISUALIZACIÓN (Función principal de Streamlit)
# =================================================================
def analizar_y_visualizar(df_patrimonio):
    """
    Procesa, analiza, audita y visualiza los datos de patrimonio neto desde un DataFrame.
    """
    st.subheader("🔬 Análisis y Reporte de Auditoría")

    # --- 1. Data Preprocessing ---
    df_patrimonio['Fecha_Inicio_Periodo'] = pd.to_datetime(df_patrimonio['Fecha_Inicio_Periodo'])
    df_patrimonio.fillna(0, inplace=True)
    st.info("Preprocesamiento de datos completo.")

    # --- 2. Verificación de Auditoría ---
    st.header("🧐 Algoritmos de Auditoría y Verificación")

    df_patrimonio['PN_Cierre_Calculado'] = (
        df_patrimonio['Patrimonio_Neto_Inicio'] + df_patrimonio['Aportes_Capital'] -
        df_patrimonio['Menos_Distribuciones_Dividendos'] + df_patrimonio['Resultado_del_Ejercicio'] +
        df_patrimonio['Otras_Variaciones']
    ).round(2)
    df_patrimonio['Discrepancia'] = df_patrimonio['PN_Cierre_Calculado'] - df_patrimonio['Patrimonio_Neto_Cierre']

    inconsistencias = df_patrimonio[df_patrimonio['Discrepancia'].abs() > 0.01]

    st.subheader("Verificación de Consistencia del Patrimonio Neto al Cierre")
    if not inconsistencias.empty:
        st.warning(f"⚠️ ¡Alerta! Se encontraron {len(inconsistencias)} inconsistencias en el cálculo.")
        st.dataframe(inconsistencias[['Fecha_Inicio_Periodo', 'Patrimonio_Neto_Cierre', 'PN_Cierre_Calculado', 'Discrepancia']])
    else:
        st.success("✅ El cálculo del Patrimonio Neto al Cierre es consistente.")

    # --- 3. Detección de Anomalías (Isolation Forest) ---
    st.subheader("🤖 Detección de Anomalías con Isolation Forest")
    features = ['Resultado_del_Ejercicio', 'Aportes_Capital', 'Menos_Distribuciones_Dividendos']
    df_ia = df_patrimonio[features].copy().fillna(0)

    iso_forest = IsolationForest(random_state=42, contamination=0.1)
    df_patrimonio['es_anomalia'] = iso_forest.fit_predict(df_ia)

    anomalias_detectadas = df_patrimonio[df_patrimonio['es_anomalia'] == -1]
    st.warning(f"Se detectaron {len(anomalias_detectadas)} períodos con variaciones anómalas.")
    if not anomalias_detectadas.empty:
        st.dataframe(anomalias_detectadas[['Fecha_Inicio_Periodo', 'Resultado_del_Ejercicio', 'Aportes_Capital', 'Menos_Distribuciones_Dividendos']])

    # --- 4. Visualizaciones ---
    st.subheader("📊 Gráficos del Análisis")
    sns.set_style("whitegrid")

    # Gráfico 1: Evolución del Patrimonio Neto
    st.write('### 1. Evolución del Patrimonio Neto')
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df_patrimonio['Fecha_Inicio_Periodo'], df_patrimonio['Patrimonio_Neto_Inicio'], marker='o', label='PN al Inicio', color='dodgerblue')
    ax1.plot(df_patrimonio['Fecha_Inicio_Periodo'], df_patrimonio['Patrimonio_Neto_Cierre'], marker='x', label='PN al Cierre', color='darkorange', linestyle='--')
    ax1.set_title('Evolución del Patrimonio Neto a lo Largo del Tiempo', fontsize=16)
    ax1.set_xlabel('Fecha de Inicio del Período', fontsize=12)
    ax1.set_ylabel('Patrimonio Neto (en ARS)', fontsize=12)
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig1)

    # Gráfico 2: Componentes de la Variación del Patrimonio Neto
    st.write('### 2. Componentes de la Variación del Patrimonio Neto')
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    df_changes = df_patrimonio.set_index('Fecha_Inicio_Periodo')[['Aportes_Capital', 'Resultado_del_Ejercicio']]
    df_changes['Menos_Distribuciones_Dividendos'] = -df_patrimonio.set_index('Fecha_Inicio_Periodo')['Menos_Distribuciones_Dividendos']
    df_changes.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
    ax2.set_title('Componentes de la Variación del Patrimonio Neto por Período', fontsize=16)
    ax2.set_xlabel('Período', fontsize=12)
    ax2.set_ylabel('Monto de la Variación (en ARS)', fontsize=12)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.legend(title='Tipo de Variación')
    st.pyplot(fig2)

    # Gráfico 3: Detección de Anomalías
    st.write('### 3. Detección de Anomalías: Resultado vs. Aportes')
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=df_patrimonio, x='Resultado_del_Ejercicio', y='Aportes_Capital',
        hue='es_anomalia', style='es_anomalia', palette={1: 'green', -1: 'red'},
        markers={1: 'o', -1: 'X'}, s=100, alpha=0.8, ax=ax3
    )
    ax3.set_title('Detección de Anomalías: Resultado del Ejercicio vs. Aportes', fontsize=16)
    ax3.set_xlabel('Resultado del Ejercicio (ARS)', fontsize=12)
    ax3.set_ylabel('Aportes de Capital (ARS)', fontsize=12)
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, ['Normal', 'Anómala'], title='¿Es Anomalía?')
    ax3.grid(True)
    st.pyplot(fig3)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Análisis de Patrimonio Neto")
    st.title('📈 Análisis de Estado de Patrimonio Neto')
    st.markdown("""
        Esta aplicación simula datos de la evolución del patrimonio neto, realiza una auditoría
        de consistencia y aplica un modelo de detección de anomalías.
    """)
    patrimonio_df = generar_dataframe_patrimonio_neto()
    analizar_y_visualizar(patrimonio_df)