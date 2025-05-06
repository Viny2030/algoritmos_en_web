import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Título de la página
st.title("Análisis de Fraude y Corrupción")

# Cargar datos
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/fraude.csv"
df = pd.read_csv(url)

st.subheader("Vista General de los Datos")
st.dataframe(df)

# Análisis de Tipos
st.subheader("Tipos de Eventos")
st.write(df['Tipo'].value_counts())

# Ejemplos de Descripciones
st.subheader("Ejemplos de Descripciones")
st.dataframe(df[['Tipo', 'Descripcion']])  # Asegúrate que la columna sea 'Descripcion'

# Análisis de Monto
st.subheader("Análisis de Montos (Impacto Económico Potencial)")
monto_sospechoso = df[df['Estado'] == 'Sospechoso']['Monto'].sum()
monto_investigado = df[df['Estado'] == 'Investigado']['Monto'].sum()
st.write("Monto Total de Eventos Sospechosos:", monto_sospechoso)
st.write("Monto Total de Eventos Investigados:", monto_investigado)

# Gráfico de eventos por tipo y estado
st.subheader("Frecuencia de Tipos de Eventos por Estado")
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.countplot(x='Tipo', hue='Estado', data=df, ax=ax1)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig1)

# Histograma de montos
st.subheader("Distribución de Montos de Eventos")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.histplot(df['Monto'], bins=5, kde=True, ax=ax2)
st.pyplot(fig2)
