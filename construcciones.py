import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# URL del dataset
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_construccion.csv"

@st.cache_data
def load_and_preprocess_data(url):
    """Carga y preprocesa los datos de construcción."""
    try:
        df_construccion = pd.read_csv(url)
    except Exception as e:
        st.error(f"Error al cargar el archivo desde la URL: {e}")
        st.info("Intentando cargar un archivo local 'df_construccion (1).csv' si existe...")
        try:
            df_construccion = pd.read_csv('df_construccion (1).csv')
        except FileNotFoundError:
            st.error("Error: No se pudo cargar el archivo CSV desde la URL ni localmente.")
            return None
        except Exception as e_local:
            st.error(f"Error al cargar el archivo local: {e_local}")
            return None

    if df_construccion is None or df_construccion.empty:
        st.error("Error: El DataFrame de construcción está vacío o no se pudo cargar.")
        return None

    # Convertir columnas de fecha a datetime
    for col in ['Fecha_Inicio', 'Fecha_Fin_Estimada']:
        if col in df_construccion.columns and df_construccion[col].dtype == 'object':
            df_construccion[col] = pd.to_datetime(df_construccion[col], errors='coerce')

    # Ingeniería de Características (sin depender de sklearn para LabelEncoder y StandardScaler)
    df_construccion['Duracion_Estimada_Dias'] = (df_construccion['Fecha_Fin_Estimada'] - df_construccion['Fecha_Inicio']).dt.days.fillna(0)
    df_construccion['Plazo_Ejecucion_Dias'] = df_construccion['Plazo_Ejecucion_Dias'].fillna(0)
    df_construccion['Relacion_Plazo'] = df_construccion['Plazo_Ejecucion_Dias'] / (df_construccion['Duracion_Estimada_Dias'] + 1e-6)
    df_construccion['Avance_Obra_Porcentaje'] = df_construccion['Avance_Obra_Porcentaje'].fillna(0)
    df_construccion['Tiempo_Transcurrido_Dias'] = (pd.Timestamp('today') - df_construccion['Fecha_Inicio']).dt.days.fillna(0)
    df_construccion['Avance_Anormal'] = np.where((df_construccion['Tiempo_Transcurrido_Dias'] > 0) &
                                                (df_construccion['Avance_Obra_Porcentaje'] / (df_construccion['Tiempo_Transcurrido_Dias'] + 1e-6) < 0.1), 1, 0)
    df_construccion['Modificaciones_Contrato'] = df_construccion['Modificaciones_Contrato'].fillna(0)
    df_construccion['Adjudicacion_Unica'] = df_construccion['Adjudicacion_Unica'].fillna(0).astype(int)
    df_construccion['Monto_Contrato'] = df_construccion['Monto_Contrato'].fillna(0)

    # Codificación simple de variables categóricas (sin LabelEncoder de sklearn)
    categorical_cols = ['Organismo_Contratante', 'Tipo_Obra', 'Estado_Obra', 'Empresa_Constructora']
    for col in categorical_cols:
        if col in df_construccion.columns and df_construccion[col].dtype == 'object':
            df_construccion[col + '_Cod'] = pd.factorize(df_construccion[col])[0]
        else:
            df_construccion[col + '_Cod'] = -1

    df_construccion['Es_Sospechoso'] = df_construccion['Es_Sospechoso'].fillna(0).astype(int)
    return df_construccion

df = load_and_preprocess_data(url)

if df is not None:
    st.title("Análisis de Posible Corrupción en Proyectos de Construcción")
    st.markdown("Este panel muestra un análisis de proyectos de construcción para identificar posibles casos de corrupción.")

    # Filtros en la barra lateral
    st.sidebar.header("Filtros")
    organismo_filter = st.sidebar.multiselect("Organismo Contratante:", df['Organismo_Contratante'].unique())
    tipo_obra_filter = st.sidebar.multiselect("Tipo de Obra:", df['Tipo_Obra'].unique())
    estado_obra_filter = st.sidebar.multiselect("Estado de Obra:", df['Estado_Obra'].unique())
    empresa_filter = st.sidebar.multiselect("Empresa Constructora:", df['Empresa_Constructora'].unique())
    sospechoso_filter = st.sidebar.checkbox("Mostrar solo proyectos sospechosos", False)

    # Aplicar filtros
    df_filtered = df.copy()
    if organismo_filter:
        df_filtered = df_filtered[df_filtered['Organismo_Contratante'].isin(organismo_filter)]
    if tipo_obra_filter:
        df_filtered = df_filtered[df_filtered['Tipo_Obra'].isin(tipo_obra_filter)]
    if estado_obra_filter:
        df_filtered = df_filtered[df_filtered['Estado_Obra'].isin(estado_obra_filter)]
    if empresa_filter:
        df_filtered = df_filtered[df_filtered['Empresa_Constructora'].isin(empresa_filter)]
    if sospechoso_filter:
        df_filtered = df_filtered[df_filtered['Es_Sospechoso'] == 1]

    st.subheader("Proyectos de Construcción Filtrados")
    st.dataframe(df_filtered)

    st.markdown("---")
    st.subheader("Análisis Exploratorio")

    # Distribución de proyectos sospechosos
    st.subheader("Distribución de Proyectos Sospechosos")
    sospechoso_counts = df_filtered['Es_Sospechoso'].value_counts()
    labels = []
    if 0 in sospechoso_counts:
        labels.append('No Sospechoso')
    if 1 in sospechoso_counts:
        labels.append('Sospechoso')

    if not sospechoso_counts.empty:
        fig_sospechoso, ax_sospechoso = plt.subplots()
        ax_sospechoso.pie(sospechoso_counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax_sospechoso.axis('equal')
        st.pyplot(fig_sospechoso)
    else:
        st.info("No hay datos disponibles para mostrar la distribución de proyectos sospechosos con los filtros actuales.")

    # Monto del contrato vs. Sospechoso
    st.subheader("Monto del Contrato vs. Sospecha")
    fig_monto, ax_monto = plt.subplots()
    sns.boxplot(x='Es_Sospechoso', y='Monto_Contrato', data=df_filtered, ax=ax_monto)
    ax_monto.set_xticks([0, 1])
    ax_monto.set_xticklabels(['No Sospechoso', 'Sospechoso'])
    ax_monto.set_xlabel("¿Es Sospechoso?")
    ax_monto.set_ylabel("Monto del Contrato")
    st.pyplot(fig_monto)

    # Relación Plazo vs. Sospechoso
    st.subheader("Relación Plazo vs. Sospecha")
    fig_plazo, ax_plazo = plt.subplots()
    sns.boxplot(x='Es_Sospechoso', y='Relacion_Plazo', data=df_filtered, ax=ax_plazo)
    ax_plazo.set_xticks([0, 1])
    ax_plazo.set_xticklabels(['No Sospechoso', 'Sospechoso'])
    ax_plazo.set_xlabel("¿Es Sospechoso?")
    ax_plazo.set_ylabel("Relación Plazo (Plazo Ejecución / Duración Estimada)")
    st.pyplot(fig_plazo)

    # Avance Anormal vs. Sospechoso
    st.subheader("Avance Anormal vs. Sospecha")
    avance_anormal_counts = df_filtered.groupby('Es_Sospechoso')['Avance_Anormal'].value_counts().unstack(fill_value=0)

    fig_avance, ax_avance = plt.subplots()

    # Asegurarse de que las columnas existen (0 y 1 para No Anormal y Anormal)
    col_no_anormal = 0 in avance_anormal_counts.columns
    col_anormal = 1 in avance_anormal_counts.columns

    # Datos para la barra 'No Anormal'
    if col_no_anormal:
        ax_avance.bar(avance_anormal_counts.index, avance_anormal_counts[0], label='No Anormal')

    # Datos para la barra 'Anormal' (apilada sobre la anterior)
    if col_anormal and col_no_anormal:
        ax_avance.bar(avance_anormal_counts.index, avance_anormal_counts[1], bottom=avance_anormal_counts[0], label='Anormal')
    elif col_anormal:
        ax_avance.bar(avance_anormal_counts.index, avance_anormal_counts[1], label='Anormal')

    ax_avance.set_xlabel("¿Es Sospechoso?")
    ax_avance.set_ylabel("Número de Proyectos")
    ax_avance.set_xticks([0, 1])
    ax_avance.set_xticklabels(['No Sospechoso', 'Sospechoso'], rotation=0)
    ax_avance.legend(title='Avance Anormal')
    st.pyplot(fig_avance)

    st.markdown("---")
    st.info("Este panel proporciona una visualización de los datos de proyectos de construcción y una indicación de posibles casos sospechosos.")