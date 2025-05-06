import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")  # Ignora advertencias innecesarias

st.title('Análisis de Casos Judiciales')

# URL corregida del dataset
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/main/df_casos_judiciales.csv"

@st.cache_data
def load_data(url):
    """Carga los datos desde la URL."""
    df = pd.read_csv(url)
    return df

# Cargar los datos
df_casos = load_data(url)

# Verificar columnas del DataFrame
st.write("Columnas disponibles en el DataFrame:", df_casos.columns.tolist())

st.subheader('Dataset de Casos Judiciales Cargado:')
st.dataframe(df_casos.head())

st.subheader('Análisis General')

# Número total de casos
st.metric("Número Total de Casos", len(df_casos))

st.subheader('Análisis del Tiempo de Proceso')

# Tiempo promedio de proceso
promedio_tiempo = df_casos['Tiempo_Proceso_Dias'].mean()
minimo_tiempo = df_casos['Tiempo_Proceso_Dias'].min()
maximo_tiempo = df_casos['Tiempo_Proceso_Dias'].max()

col1, col2, col3 = st.columns(3)
col1.metric("Tiempo Promedio de Proceso (días)", f"{promedio_tiempo:.2f}")
col2.metric("Tiempo Mínimo de Proceso (días)", minimo_tiempo)
col3.metric("Tiempo Máximo de Proceso (días)", maximo_tiempo)

st.markdown("#### Distribución del Tiempo de Proceso")
fig_tiempo, ax_tiempo = plt.subplots(figsize=(10, 6))
sns.histplot(df_casos['Tiempo_Proceso_Dias'], bins=20, kde=True, ax=ax_tiempo)
ax_tiempo.set_title('Distribución del Tiempo de Proceso (en días)')
ax_tiempo.set_xlabel('Tiempo de Proceso (Días)')
ax_tiempo.set_ylabel('Frecuencia')
st.pyplot(fig_tiempo)

st.subheader('Análisis de Fallos')

fallos_counts = df_casos['Fallo'].value_counts()
st.markdown("#### Distribución de los Fallos")
fig_fallos, ax_fallos = plt.subplots(figsize=(6, 6))
ax_fallos.pie(fallos_counts, labels=fallos_counts.index, autopct='%1.1f%%', startangle=140)
ax_fallos.axis('equal')
st.pyplot(fig_fallos)
st.write("Distribución de los Fallos:", fallos_counts)

st.subheader('Tiempo de Proceso por Tipo de Causa')

tiempo_por_causa = df_casos.groupby('Tipo_Causa')['Tiempo_Proceso_Dias'].mean().sort_values(ascending=False)
st.markdown("#### Tiempo Promedio de Proceso por Tipo de Causa")
fig_tiempo_causa, ax_tiempo_causa = plt.subplots(figsize=(8, 6))
tiempo_por_causa.plot(kind='bar', ax=ax_tiempo_causa)
ax_tiempo_causa.set_title('Tiempo Promedio de Proceso por Tipo de Causa')
ax_tiempo_causa.set_ylabel('Tiempo Promedio (Días)')
ax_tiempo_causa.set_xlabel('Tipo de Causa')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_tiempo_causa)
st.write("Tiempo promedio de proceso por tipo de causa (días):", tiempo_por_causa)

st.subheader('Fallos por Tipo de Causa')

fallos_por_causa = df_casos.groupby(['Tipo_Causa', 'Fallo']).size().unstack(fill_value=0)
st.markdown("#### Distribución de Fallos por Tipo de Causa")
fig_fallos_causa, ax_fallos_causa = plt.subplots(figsize=(10, 6))
fallos_por_causa.plot(kind='bar', stacked=True, ax=ax_fallos_causa)
ax_fallos_causa.set_title('Distribución de Fallos por Tipo de Causa')
ax_fallos_causa.set_ylabel('Número de Casos')
ax_fallos_causa.set_xlabel('Tipo de Causa')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_fallos_causa)
st.write("Distribución de los Fallos por Tipo de Causa:", fallos_por_causa)

st.subheader('Tiempo de Proceso por Estado Actual')

tiempo_por_estado = df_casos.groupby('Estado_Actual')['Tiempo_Proceso_Dias'].mean().sort_values(ascending=False)
st.markdown("#### Tiempo Promedio de Proceso por Estado Actual")
fig_tiempo_estado, ax_tiempo_estado = plt.subplots(figsize=(8, 6))
tiempo_por_estado.plot(kind='bar', ax=ax_tiempo_estado)
ax_tiempo_estado.set_title('Tiempo Promedio de Proceso por Estado Actual')
ax_tiempo_estado.set_ylabel('Tiempo Promedio (Días)')
ax_tiempo_estado.set_xlabel('Estado Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_tiempo_estado)
st.write("Tiempo promedio de proceso por estado actual (días):", tiempo_por_estado)

st.subheader('Fallos por Instancia')

fallos_por_instancia = df_casos.groupby(['Instancia', 'Fallo']).size().unstack(fill_value=0)
st.markdown("#### Distribución de Fallos por Instancia")
fig_fallos_instancia, ax_fallos_instancia = plt.subplots(figsize=(8, 6))
fallos_por_instancia.plot(kind='bar', stacked=True, ax=ax_fallos_instancia)
ax_fallos_instancia.set_title('Distribución de Fallos por Instancia')
ax_fallos_instancia.set_ylabel('Número de Casos')
ax_fallos_instancia.set_xlabel('Instancia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_fallos_instancia)
st.write("Distribución de los Fallos por Instancia:", fallos_por_instancia)

