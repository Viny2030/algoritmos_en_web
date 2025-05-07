import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker
import plotly.express as px
import re

# 1. Generación de Datos Simulados para Medicamentos (Streamlit puede recargar, así que cargamos una vez)
@st.cache_data
def load_medicamentos_data(url):
    fake = Faker('es_AR')
    df_medicamentos = pd.read_csv(url)
    df_medicamentos['Precio_Farmacia_B'] = df_medicamentos['Precio_Farmacia_B'].apply(lambda x: max(0, x))
    df_medicamentos['Precio_Farmacia_C'] = df_medicamentos['Precio_Farmacia_C'].apply(lambda x: max(0, x))
    return df_medicamentos

url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_medicamentos.csv"
df_medicamentos = load_medicamentos_data(url)

# 2. Análisis de Precios y Comparación con Porcentajes de Diferencia
def analizar_diferencia_porcentual(row, umbral_significativo=25):
    precios = {'A': row['Precio_Farmacia_A'],
               'B': row['Precio_Farmacia_B'],
               'C': row['Precio_Farmacia_C']}
    disponibilidades = {'A': row['Disponibilidad_Farmacia_A'],
                          'B': row['Disponibilidad_Farmacia_B'],
                          'C': row['Disponibilidad_Farmacia_C']}
    precios_disponibles = {farmacia: precio for farmacia, precio in precios.items() if disponibilidades[farmacia] == 'Sí'}

    if not precios_disponibles:
        return 'Sin disponibilidad'
    elif len(precios_disponibles) == 1:
        farmacia, precio = list(precios_disponibles.items())[0]
        return f'Disponible solo en {farmacia} a ${precio:.2f}'
    else:
        min_precio = min(precios_disponibles.values())
        farmacias_min = [f for f, p in precios_disponibles.items() if p == min_precio]
        max_precio = max(precios_disponibles.values())
        farmacias_max = [f for f, p in precios_disponibles.items() if p == max_precio]

        diferencia_porcentual = 0
        if min_precio > 0:
            diferencia_porcentual = ((max_precio - min_precio) / min_precio) * 100

        es_significativa = "Sí" if diferencia_porcentual > umbral_significativo else "No"

        return f'Mínimo: {", ".join(farmacias_min)} (${min_precio:.2f}), Máximo: {", ".join(farmacias_max)} (${max_precio:.2f}), Diferencia: {diferencia_porcentual:.2f}%, Significativa: {es_significativa}'

df_medicamentos['Analisis_Precio'] = df_medicamentos.apply(analizar_diferencia_porcentual, axis=1)

# 3. Identificación de Medicamentos con Diferencias Significativas de Precio (Basado en porcentaje)
def es_diferencia_significativa_porcentual(row, umbral_porcentaje=25):
    precios = [row['Precio_Farmacia_A'], row['Precio_Farmacia_B'], row['Precio_Farmacia_C']]
    precios_validos = [p for p in precios if p > 0]
    if not precios_validos or len(precios_validos) < 2:
        return 0
    min_precio = min(precios_validos)
    max_precio = max(precios_validos)
    diferencia_porcentual = ((max_precio - min_precio) / min_precio) * 100
    return 1 if diferencia_porcentual > umbral_porcentaje else 0

df_medicamentos['Diferencia_Significativa_%'] = df_medicamentos.apply(es_diferencia_significativa_porcentual, axis=1, umbral_porcentaje=20)

# --- INTERFAZ DE STREAMLIT ---
st.title("Análisis de Precios de Medicamentos")
st.markdown("Explora las diferencias de precios de medicamentos entre farmacias.")

# Mostrar el DataFrame con el análisis
st.subheader("Tabla de Análisis de Precios")
st.dataframe(df_medicamentos[['Nombre_Medicamento', 'Principio_Activo', 'Laboratorio',
                                'Precio_Farmacia_A', 'Precio_Farmacia_B', 'Precio_Farmacia_C',
                                'Disponibilidad_Farmacia_A', 'Disponibilidad_Farmacia_B', 'Disponibilidad_Farmacia_C',
                                'Analisis_Precio', 'Diferencia_Significativa_%']])

st.markdown("---")
st.subheader("Visualización de Diferencias de Precio Significativas")

# Filtrar medicamentos con diferencias significativas
medicamentos_significativos = df_medicamentos[df_medicamentos['Diferencia_Significativa_%'] == 1]

if not medicamentos_significativos.empty:
    # Gráfico de barras interactivo con Plotly
    st.subheader("Precios de Medicamentos con Diferencias Significativas (Interactivo)")
    melted_df = medicamentos_significativos.melt(
        id_vars=['Nombre_Medicamento'],
        value_vars=['Precio_Farmacia_A', 'Precio_Farmacia_B', 'Precio_Farmacia_C'],
        var_name='Farmacia',
        value_name='Precio'
    )
    fig_precios_interactivo = px.bar(
        melted_df,
        x='Nombre_Medicamento',
        y='Precio',
        color='Farmacia',
        title='Precios por Farmacia (Medicamentos con Diferencias Significativas)',
        labels={'Nombre_Medicamento': 'Medicamento', 'Precio': 'Precio', 'Farmacia': 'Farmacia'},
        barmode='group'
    )
    st.plotly_chart(fig_precios_interactivo)

else:
    st.info("No hay medicamentos con diferencias de precio significativas para mostrar en el gráfico.")

st.markdown("---")
st.subheader("Distribución de Diferencias de Precio")

# Crear un histograma interactivo de las diferencias porcentuales significativas con Plotly
diferencias_significativas_str = df_medicamentos[df_medicamentos['Diferencia_Significativa_%'] == 1]['Analisis_Precio'].str.extract(r'Diferencia: (\d+\.\d+)%')
diferencias_significativas = pd.to_numeric(diferencias_significativas_str[0], errors='coerce').dropna()

if not diferencias_significativas.empty:
    st.subheader("Distribución de Diferencias de Precio Significativas (Interactivo)")
    fig_hist_interactivo = px.histogram(
        diferencias_significativas,
        x=diferencias_significativas,
        nbins=10,
        title='Distribución de Diferencias de Precio Significativas (%)',
        labels={'x': 'Diferencia Porcentual (%)', 'count': 'Número de Medicamentos'}
    )
    st.plotly_chart(fig_hist_interactivo)
else:
    st.info("No hay diferencias de precio significativas para mostrar en el histograma.")

st.markdown("---")
st.subheader("Análisis de Disponibilidad y Precio Mínimo")

# Crear un gráfico de dispersión interactivo de Precio Mínimo vs. Disponibilidad
def obtener_precio_minimo(analisis):
    if 'Mínimo:' in analisis:
        match = re.search(r'\$\s*(\d+\.\d+)', analisis)
        if match:
            return float(match.group(1))
    return np.nan

df_medicamentos['Precio_Minimo'] = df_medicamentos['Analisis_Precio'].apply(obtener_precio_minimo)

# Crear una columna combinada de disponibilidad
def obtener_disponibilidad_combinada(row):
    disponibilidades = [row['Disponibilidad_Farmacia_A'], row['Disponibilidad_Farmacia_B'], row['Disponibilidad_Farmacia_C']]
    if all(d == 'No' for d in disponibilidades):
        return 'No Disponible'
    elif any(d == 'Sí' for d in disponibilidades):
        return 'Disponible'
    else:
        return 'Inconsistente'

df_medicamentos['Disponibilidad_Combinada'] = df_medicamentos.apply(obtener_disponibilidad_combinada, axis=1)

# Gráfico de dispersión interactivo
if not df_medicamentos['Precio_Minimo'].isnull().all():  # Verificar si hay datos válidos
    st.subheader("Precio Mínimo vs. Disponibilidad (Interactivo)")
    fig_dispersion = px.scatter(
        df_medicamentos,
        x='Precio_Minimo',
        y='Diferencia_Significativa_%',
        color='Disponibilidad_Combinada',
        title='Precio Mínimo de Medicamento vs. Diferencia de Precio Significativa y Disponibilidad',
        labels={'Precio_Minimo': 'Precio Mínimo', 'Diferencia_Significativa_%': 'Diferencia Significativa (%)', 'Disponibilidad_Combinada': 'Disponibilidad'},
        hover_data=['Nombre_Medicamento', 'Laboratorio']
    )
    st.plotly_chart(fig_dispersion)
else:
    st.info("No hay datos suficientes para mostrar el gráfico de Precio Mínimo vs. Disponibilidad.")

st.markdown("---")
st.info("Este panel muestra un análisis de las diferencias de precios de medicamentos entre farmacias y visualiza aquellos con diferencias significativas, así como su disponibilidad.")