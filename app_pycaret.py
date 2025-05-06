import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, pull, save_model, load_model
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import numpy as np
import matplotlib.pyplot as plt

st.title('Análisis de Fraude en E-commerce con PyCaret')

# Obtén la ruta absoluta del directorio actual
try:
    current_directory = os.path.abspath(os.path.dirname(__file__))  # Usamos __file__ aquí
except NameError:
    current_directory = os.getcwd()  # Si __file__ no está definido, usamos el CWD

# 1. Cargar el dataset df_ecommerce.csv
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_ecommerce.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df['Fecha_Hora'] = pd.to_datetime(df['Fecha_Hora'])
    df['Hora'] = df['Fecha_Hora'].dt.hour
    df['Dia_Semana'] = df['Fecha_Hora'].dt.dayofweek
    return df

try:
    df_ecommerce = load_data(url)
    st.subheader('Dataset Cargado:')
    st.dataframe(df_ecommerce.head())

    # --- Sección de Análisis de Transacciones Fraudulentas con PyCaret ---
    st.subheader('Análisis de Transacciones Fraudulentas con PyCaret')
    if st.checkbox('Mostrar análisis de transacciones fraudulentas con PyCaret'):
        st.markdown("### Identificación de Transacciones Fraudulentas (PyCaret)")
        if 'Monto' in df_ecommerce.columns and 'Hora' in df_ecommerce.columns and 'Dia_Semana' in df_ecommerce.columns and 'Producto' in df_ecommerce.columns and 'Es_Fraudulenta' in df_ecommerce.columns:
            data_prep = df_ecommerce[['Monto', 'Hora', 'Dia_Semana', 'Producto', 'Es_Fraudulenta']].copy()

            # Configuración de PyCaret
            st.subheader("Configuración de PyCaret")
            with st.spinner('Configurando el entorno de PyCaret...'):
                s = setup(data_prep, target='Es_Fraudulenta', session_id=42,
                          categorical_features=['Producto'],
                          numeric_features=['Monto', 'Hora', 'Dia_Semana'],
                          train_size=0.7,
                          transformation=True,
                          normalize=True,
                          remove_multicollinearity=True, multicollinearity_threshold=0.9,
                          feature_selection=True, feature_selection_threshold=0.8)
                st.success('Entorno de PyCaret configurado exitosamente!')

            # Comparación de modelos
            st.subheader("Comparación de Modelos")
            if st.button('Comparar Modelos de Clasificación'):
                with st.spinner('Entrenando y comparando modelos...'):
                    best_model = compare_models()
                    compare_df = pull()
                    st.dataframe(compare_df)
                    st.success('Modelos comparados!')

                    # Mostrar el mejor modelo
                    st.subheader("Mejor Modelo Seleccionado:")
                    st.write(best_model)

                    # Guardar el mejor modelo
                    save_model(best_model, 'best_fraud_model')
                    st.info('El mejor modelo ha sido guardado como best_fraud_model.pkl')

            # Cargar el mejor modelo y hacer predicciones (opcional)
            if os.path.exists('best_fraud_model.pkl'):
                if st.checkbox('Cargar el mejor modelo guardado y hacer predicciones'):
                    loaded_best_model = load_model('best_fraud_model')
                    new_data = data_prep.drop('Es_Fraudulenta', axis=1).sample(10, random_state=42)
                    predictions = predict_model(loaded_best_model, data=new_data)
                    st.subheader("Predicciones con el Mejor Modelo:")
                    st.dataframe(predictions)

        else:
            st.warning("No se pueden realizar análisis de transacciones fraudulentas con PyCaret debido a la falta de columnas necesarias.")

    # --- Sección de Detección de Cuentas Falsas ---
    st.subheader('Detección de Cuentas Falsas y Actividades Maliciosas')
    if st.checkbox('Mostrar detección de cuentas falsas'):
        st.markdown("### Detección con Agrupamiento (DBSCAN en IPs)")
        if 'Direccion_IP' in df_ecommerce.columns:
            le_ip = LabelEncoder()
            df_ecommerce['IP_Codificada'] = le_ip.fit_transform(df_ecommerce['Direccion_IP'])
            ip_array = df_ecommerce[['IP_Codificada']].values
            scaler_ip = StandardScaler()
            ip_scaled = scaler_ip.fit_transform(ip_array)

            # Ajustar los parámetros de DBSCAN con sliders
            eps_dbscan_ip = st.slider('Epsilon (eps) para DBSCAN (IP)', 0.1, 1.0, 0.5, 0.05)
            min_samples_dbscan_ip = st.slider('Min Samples para DBSCAN (IP)', 2, 10, 2)

            dbscan_ip = DBSCAN(eps=eps_dbscan_ip, min_samples=min_samples_dbscan_ip)
            df_ecommerce['Grupo_IP'] = dbscan_ip.fit_predict(ip_scaled)

            st.subheader('Agrupamiento de IPs (DBSCAN):')
            st.dataframe(df_ecommerce[['ID_Usuario', 'Direccion_IP', 'Grupo_IP']].head())
            st.write("Grupos de IP (Clusters):", np.unique(df_ecommerce['Grupo_IP']))

            grupos_sospechosos_ip = df_ecommerce.groupby('Grupo_IP')['Es_Fraudulenta'].sum().sort_values(ascending=False)
            st.subheader('Número de Transacciones Fraudulentas por Grupo de IP:')
            st.dataframe(grupos_sospechosos_ip)

            # Visualización de los clusters de IP
            fig_ip, ax_ip = plt.subplots()
            scatter_ip = ax_ip.scatter(ip_scaled[:, 0], np.zeros(len(ip_scaled)), c=df_ecommerce['Grupo_IP'], cmap='viridis')
            ax_ip.set_xlabel("IP Codificada (Escalada)")
            ax_ip.set_yticks([])
            ax_ip.set_title("Clusters de Direcciones IP (DBSCAN)")
            legend_ip = ax_ip.legend(*scatter_ip.legend_elements(), title="Grupo IP")
            ax_ip.add_artist(legend_ip)
            st.pyplot(fig_ip)

        else:
            st.warning("No se puede realizar el análisis de detección de cuentas falsas porque falta la columna 'Direccion_IP'.")

    # --- Sección de Análisis de Comportamiento del Usuario ---
    st.subheader('Análisis de Comportamiento del Usuario')
    if st.checkbox('Mostrar análisis de comportamiento del usuario'):
        st.markdown("### Detección de Fraude Basado en Frecuencia de Transacciones")
        if 'ID_Usuario' in df_ecommerce.columns and 'Fecha_Hora' in df_ecommerce.columns:
            frecuencia_usuarios = df_ecommerce.groupby('ID_Usuario')['Fecha_Hora'].count().reset_index(name='Num_Transacciones')
            st.subheader('Frecuencia de Transacciones por Usuario:')
            st.dataframe(frecuencia_usuarios.head())

            umbral_frecuencia = st.slider('Umbral de Frecuencia de Transacciones para Alta Actividad', 1, 10, 3)
            usuarios_alta_actividad = frecuencia_usuarios[frecuencia_usuarios['Num_Transacciones'] > umbral_frecuencia]['ID_Usuario'].tolist()

            if usuarios_alta_actividad:
                st.warning(f"Usuarios con Alta Actividad (>{umbral_frecuencia} transacciones): {usuarios_alta_actividad[:10]}...")
                transacciones_alta_actividad = df_ecommerce[df_ecommerce['ID_Usuario'].isin(usuarios_alta_actividad)]
                st.subheader('Primeras Transacciones de Usuarios con Alta Actividad:')
                st.dataframe(transacciones_alta_actividad[['ID_Usuario', 'Fecha_Hora', 'Monto', 'Es_Fraudulenta']].head())

                # Visualización de la distribución de frecuencia
                fig_freq, ax_freq = plt.subplots()
                ax_freq.hist(frecuencia_usuarios['Num_Transacciones'], bins=20, edgecolor='black')
                ax_freq.axvline(umbral_frecuencia, color='red', linestyle='dashed', linewidth=2, label=f'Umbral: {umbral_frecuencia}')
                ax_freq.set_xlabel('Número de Transacciones por Usuario')
                ax_freq.set_ylabel('Número de Usuarios')
                ax_freq.set_title('Distribución de Frecuencia de Transacciones')
                ax_freq.legend()
                st.pyplot(fig_freq)

            else:
                st.info("No se encontraron usuarios con una alta frecuencia de transacciones según el umbral seleccionado.")
        else:
            st.warning("No se puede realizar el análisis de comportamiento del usuario porque faltan las columnas 'ID_Usuario' o 'Fecha_Hora'.")

except Exception as e:
    st.error(f"Ocurrió un error al cargar o procesar los datos: {e}")
