import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker

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
    # Preparar datos para el gráfico
    medicamentos = medicamentos_significativos['Nombre_Medicamento'].tolist()
    precios_a = medicamentos_significativos['Precio_Farmacia_A'].tolist()
    precios_b = medicamentos_significativos['Precio_Farmacia_B'].tolist()
    precios_c = medicamentos_significativos['Precio_Farmacia_C'].tolist()

    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    index = np.arange(len(medicamentos))

    bar_a = ax.bar(index, precios_a, bar_width, label='Farmacia A')
    bar_b = ax.bar(index + bar_width, precios_b, bar_width, label='Farmacia B')
    bar_c = ax.bar(index + 2 * bar_width, precios_c, bar_width, label='Farmacia C')

    ax.set_xlabel('Medicamento')
    ax.set_ylabel('Precio')
    ax.set_title('Precios de Medicamentos con Diferencias Significativas')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(medicamentos, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("No hay medicamentos con diferencias de precio significativas para mostrar en el gráfico.")

st.markdown("---")
st.subheader("Distribución de Diferencias de Precio")

# Crear un histograma de las diferencias porcentuales significativas
diferencias_significativas = df_medicamentos[df_medicamentos['Diferencia_Significativa_%'] == 1]['Analisis_Precio'].str.extract(r'Diferencia: (\d+\.\d+)%')

if diferencias_significativas.dropna().shape[0] > 0:
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(diferencias_significativas.dropna().astype(float), bins=10, kde=True, ax=ax_hist)
    ax_hist.set_xlabel('Diferencia Porcentual (%)')
    ax_hist.set_ylabel('Número de Medicamentos')
    ax_hist.set_title('Distribución de Diferencias de Precio Significativas')
    st.pyplot(fig_hist)
else:
    st.info("No hay diferencias de precio significativas para mostrar en el histograma.")

st.markdown("---")
st.info("Este panel muestra un análisis de las diferencias de precios de medicamentos entre farmacias y visualiza aquellos con diferencias significativas.")