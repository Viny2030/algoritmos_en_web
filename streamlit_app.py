
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import DBSCAN
import os
import numpy as np
import matplotlib.pyplot as plt

st.title('Análisis de Fraude en E-commerce')

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
    return df

try:
    df_ecommerce = load_data(url)
    st.subheader('Dataset Cargado:')
    st.dataframe(df_ecommerce.head())

    # --- Sección de Análisis de Transacciones Fraudulentas ---
    st.subheader('Análisis de Transacciones Fraudulentas')
    if st.checkbox('Mostrar análisis de transacciones fraudulentas'):
        st.markdown("### Identificación de Transacciones Fraudulentas (Regresión Logística)")
        if 'Monto' in df_ecommerce.columns and 'Fecha_Hora' in df_ecommerce.columns and 'Es_Fraudulenta' in df_ecommerce.columns:
            df_ecommerce['Hora'] = df_ecommerce['Fecha_Hora'].dt.hour
            df_ecommerce['Dia_Semana'] = df_ecommerce['Fecha_Hora'].dt.dayofweek

            # Codificación de variables categóricas (manteniendo solo las más relevantes para el ejemplo)
            df_encoded_transacciones = pd.get_dummies(df_ecommerce, columns=['Producto'], prefix='Prod', dummy_na=False)

            features_transacciones = ['Monto', 'Hora', 'Dia_Semana'] + [col for col in df_encoded_transacciones.columns if col.startswith('Prod_')]
            features_transacciones = [col for col in features_transacciones if col in df_encoded_transacciones.columns]

            if 'Es_Fraudulenta' in df_encoded_transacciones.columns and all(feature in df_encoded_transacciones.columns for feature in features_transacciones):
                X_trans = df_encoded_transacciones[features_transacciones]
                y_trans = df_encoded_transacciones['Es_Fraudulenta']
                X_train_trans, X_test_trans, y_train_trans, y_test_trans = train_test_split(X_trans, y_trans, test_size=0.3, random_state=42, stratify=y_trans)

                scaler_trans = StandardScaler()
                X_train_scaled_trans = scaler_trans.fit_transform(X_train_trans)
                X_test_scaled_trans = scaler_trans.transform(X_test_trans)

                model_ecommerce = LogisticRegression(random_state=42)
                model_ecommerce.fit(X_train_scaled_trans, y_train_trans)
                y_pred_ecommerce = model_ecommerce.predict(X_test_scaled_trans)

                st.subheader('Resultados de Regresión Logística:')
                st.write("Precisión del Modelo:", accuracy_score(y_test_trans, y_pred_ecommerce))
                st.text("Reporte de Clasificación:\n" + classification_report(y_test_trans, y_pred_ecommerce, target_names=['No Fraudulenta', 'Fraudulenta'], zero_division=0))

                # Visualización de las predicciones
                if X_test_scaled_trans.shape[1] >= 2:
                    fig_trans, ax_trans = plt.subplots()
                    scatter_trans = ax_trans.scatter(X_test_scaled_trans[:, 0], X_test_scaled_trans[:, 1], c=y_pred_ecommerce, cmap='coolwarm', alpha=0.7)
                    ax_trans.set_xlabel("Feature 1 (Scaled)")
                    ax_trans.set_ylabel("Feature 2 (Scaled)")
                    ax_trans.set_title("Predicciones de Fraude (Regresión Logística)")
                    legend_trans = ax_trans.legend(*scatter_trans.legend_elements(), title="Clase Predicha")
                    ax_trans.add_artist(legend_trans)
                    st.pyplot(fig_trans)
                else:
                    st.warning("No se pueden mostrar gráficos de dispersión con menos de 2 características.")

            else:
                st.warning("No se pueden realizar análisis de transacciones fraudulentas debido a la falta de columnas necesarias.")
        else:
            st.warning("No se pueden realizar análisis de transacciones fraudulentas debido a la falta de columnas necesarias ('Monto', 'Fecha_Hora', 'Es_Fraudulenta').")

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
