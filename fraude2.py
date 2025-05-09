import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

def analizar_fraude(uploaded_file):
    try:
        if uploaded_file is not None:
            # Leer archivo
            file_name = uploaded_file.name
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Formato no soportado. Por favor, sube un archivo CSV o Excel.")
                return None, None, None, None, None

            st.subheader("Vista de Datos")
            st.dataframe(df.head())

            # Tipo de eventos
            st.subheader("Tipos de Eventos")
            if 'Tipo' in df.columns:
                tipos_eventos = df['Tipo'].value_counts().reset_index()
                tipos_eventos.columns = ['Tipo', 'Cantidad']
                st.dataframe(tipos_eventos)
            else:
                st.warning("La columna 'Tipo' no se encuentra en el archivo.")

            # Análisis de montos
            st.subheader("Montos Sospechosos/Investigados")
            if 'Monto' in df.columns and 'Estado' in df.columns:
                monto_sospechoso = df[df['Estado'] == 'Sospechoso']['Monto'].sum()
                monto_investigado = df[df['Estado'] == 'Investigado']['Monto'].sum()
                resumen_montos = pd.DataFrame({
                    'Estado': ['Sospechoso', 'Investigado'],
                    'Monto Total': [monto_sospechoso, monto_investigado]
                })
                st.dataframe(resumen_montos)
            else:
                st.warning("Las columnas 'Monto' o 'Estado' no se encuentran en el archivo.")

            # Gráfico de tipos por estado
            st.subheader("Frecuencia de Tipos de Eventos por Estado")
            if 'Tipo' in df.columns and 'Estado' in df.columns:
                fig1, ax1 = plt.subplots(figsize=(8, 4))
                sns.countplot(x='Tipo', hue='Estado', data=df, ax=ax1)
                plt.xticks(rotation=45, ha='right')
                plt.title("Frecuencia de Tipos de Eventos por Estado")
                st.pyplot(fig1)
            else:
                st.warning("No se pueden generar gráficos de tipos por estado sin las columnas 'Tipo' y 'Estado'.")

            # Histograma de montos
            st.subheader("Distribución de Montos")
            if 'Monto' in df.columns:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.histplot(df['Monto'], bins=10, kde=True, ax=ax2)
                plt.title("Distribución de Montos")
                st.pyplot(fig2)
            else:
                st.warning("No se puede generar el histograma de montos sin la columna 'Monto'.")

    except Exception as e:
        st.error(f"Ocurrió un error durante el análisis: {e}")

st.title("Análisis de Fraude y Corrupción")
st.subheader("Sube tu archivo CSV o Excel para analizar posibles fraudes.")

uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    analizar_fraude(uploaded_file)