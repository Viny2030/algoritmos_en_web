import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.title('Análisis de Fraude en Tarjetas de Crédito')

# 1. Cargar los datos
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/datos_tarjetas.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

df_tarjetas = load_data(url)

st.subheader('Dataset Cargado:')
st.dataframe(df_tarjetas.head())

# 2. Análisis Exploratorio de Datos
st.subheader('Resumen Estadístico')
st.dataframe(df_tarjetas.describe())

# 3. Distribución de transacciones fraudulentas vs normales
st.subheader('Distribución de Fraudes')
fraudes = df_tarjetas['Es_Fraude'].value_counts()
st.bar_chart(fraudes)
st.write(f"Porcentaje de fraudes: {(fraudes[1] / len(df_tarjetas)) * 100:.2f}%")

# 4. Análisis por país
st.subheader('Análisis por País')
pais_fraude = df_tarjetas.groupby(['Pais', 'Es_Fraude']).size().unstack(fill_value=0)
st.bar_chart(pais_fraude)

# 5. Análisis de montos
st.subheader('Estadísticas de Montos por Tipo de Transacción')
stats_monto = df_tarjetas.groupby('Es_Fraude')['Monto'].agg(['mean', 'std', 'min', 'max'])
st.dataframe(stats_monto)

# 6. Visualizaciones
st.subheader('Visualizaciones')

# 6.1 Distribución de montos (Boxplot)
st.subheader('Distribución de Montos por Tipo de Transacción (Boxplot)')
fig_boxplot, ax_boxplot = plt.subplots()
sns.boxplot(x='Es_Fraude', y='Monto', data=df_tarjetas, ax=ax_boxplot)
ax_boxplot.set_title('Distribución de Montos por Tipo')
ax_boxplot.set_xlabel('Es Fraude (0=No, 1=Sí)')
ax_boxplot.set_ylabel('Monto')
st.pyplot(fig_boxplot)

# 6.2 Fraudes por país (Bar Chart)
st.subheader('Número de Fraudes por País (Gráfico de Barras)')
fig_pais_fraude, ax_pais_fraude = plt.subplots()
pais_fraude[1].plot(kind='bar', ax=ax_pais_fraude)
ax_pais_fraude.set_title('Número de Fraudes por País')
ax_pais_fraude.set_xlabel('País')
ax_pais_fraude.set_ylabel('Cantidad de Fraudes')
plt.xticks(rotation=45)
st.pyplot(fig_pais_fraude)

# 6.3 Distribución de montos (Histograma)
st.subheader('Distribución de Montos (Histograma)')
fig_hist_monto, ax_hist_monto = plt.subplots()
ax_hist_monto.hist(df_tarjetas[df_tarjetas['Es_Fraude'] == 0]['Monto'],
                 bins=50, alpha=0.5, label='Normal', density=True)
ax_hist_monto.hist(df_tarjetas[df_tarjetas['Es_Fraude'] == 1]['Monto'],
                 bins=50, alpha=0.5, label='Fraude', density=True)
ax_hist_monto.legend()
ax_hist_monto.set_title('Distribución de Montos')
ax_hist_monto.set_xlabel('Monto')
ax_hist_monto.set_ylabel('Densidad')
st.pyplot(fig_hist_monto)

# 6.4 Proporción de fraudes (Pie Chart)
st.subheader('Proporción de Transacciones Fraudulentas (Gráfico de Torta)')
fig_pie_fraude, ax_pie_fraude = plt.subplots()
ax_pie_fraude.pie(fraudes, labels=fraudes.index, autopct='%1.1f%%')
ax_pie_fraude.set_title('Proporción de Transacciones Fraudulentas')
st.pyplot(fig_pie_fraude)

# 7. Implementación simple de detección de anomalías basada en umbrales
st.subheader('Detección de Anomalías Basada en Umbrales')

umbral_monto = st.slider('Seleccione el umbral de monto para detectar anomalías', 100, 10000, 5000)

def detectar_anomalias(df, umbral_monto_seleccionado):
    """
    Detecta posibles transacciones fraudulentas basadas en un umbral de monto
    """
    sospechosas = df[df['Monto'] > umbral_monto_seleccionado]
    return sospechosas

transacciones_sospechosas = detectar_anomalias(df_tarjetas, umbral_monto)

st.subheader('Transacciones Sospechosas Basadas en el Umbral Seleccionado')
st.write(f"Umbral de monto seleccionado: {umbral_monto}")
st.write(f"Número de transacciones sospechosas: {len(transacciones_sospechosas)}")
if not transacciones_sospechosas.empty:
    st.dataframe(transacciones_sospechosas[['Monto', 'Pais', 'Es_Fraude']].head())
else:
    st.info("No se encontraron transacciones sospechosas con el umbral seleccionado.")

# 8. Métricas simples de evaluación (si hay transacciones sospechosas)
if not transacciones_sospechosas.empty:
    precision = (transacciones_sospechosas['Es_Fraude'] == 1).mean()
    st.subheader('Métricas Simples de Evaluación (Basado en el Umbral)')
    st.write(f"Precisión del modelo simple (basado en el umbral): {precision:.2%}")
else:
    st.info("No se pueden calcular las métricas ya que no hay transacciones sospechosas.")