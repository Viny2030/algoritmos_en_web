import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🔎 Análisis de Contrataciones Públicas y Detección de Irregularidades")

# 1. Cargar datos desde GitHub
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/main/df_contrataciones.csv"
st.markdown("Cargando datos desde GitHub...")
try:
    df = pd.read_csv(url)
    st.success("Datos cargados correctamente.")
except Exception as e:
    st.error(f"Error al leer el archivo CSV: {e}")
    st.stop()

# 2. Preprocesamiento
# Convertir fechas
for col in ['Fecha_Publicacion', 'Fecha_Adjudicacion']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Indicadores
df['Sin_Justificacion_CD'] = np.where((df.get('Proceso_Licitacion') == 'Contratación Directa') & 
                                      (df.get('Justificacion_Contratacion_Directa').isnull() | 
                                       (df.get('Justificacion_Contratacion_Directa') == '')), 1, 0)

df['Pocas_Ofertas_Alto_Monto'] = np.where((df.get('Proceso_Licitacion') == 'Licitación Pública') &
                                          (df.get('Monto_Adjudicado') > 100000) &
                                          (df.get('Cantidad_Ofertas') <= 2), 1, 0)

df['Plazo_Corto_Obras'] = 0
if 'Tipo_Contratacion' in df.columns and 'Plazo_Ejecucion_Dias' in df.columns:
    obras = df[df['Tipo_Contratacion'] == 'Obras']
    df.loc[obras.index, 'Plazo_Corto_Obras'] = np.where(obras['Plazo_Ejecucion_Dias'] < 30, 1, 0)

proveedores_frec = df['Proveedor'].value_counts()
sospechosos = proveedores_frec[proveedores_frec > 5].index.tolist()
df['Proveedor_Frecuente'] = df['Proveedor'].apply(lambda x: 1 if x in sospechosos else 0)

df['Dias_Publicacion_Adjudicacion'] = (df['Fecha_Adjudicacion'] - df['Fecha_Publicacion']).dt.days
df['Adjudicacion_Rapida'] = np.where(df['Dias_Publicacion_Adjudicacion'] < 7, 1, 0)

# Indicador combinado
df['Sospecha_Irregularidad'] = (df[['Sin_Justificacion_CD', 'Pocas_Ofertas_Alto_Monto',
                                    'Plazo_Corto_Obras', 'Proveedor_Frecuente',
                                    'Adjudicacion_Rapida']].sum(axis=1) > 0)

# Sidebar con filtros
st.sidebar.header("Filtros")
organismo = st.sidebar.multiselect("Organismo Contratante", df['Organismo_Contratante'].dropna().unique())
proveedor = st.sidebar.multiselect("Proveedor", df['Proveedor'].dropna().unique())
solo_sospechosos = st.sidebar.checkbox("Mostrar solo contratos con posibles irregularidades", value=True)

df_filtrado = df.copy()
if organismo:
    df_filtrado = df_filtrado[df_filtrado['Organismo_Contratante'].isin(organismo)]
if proveedor:
    df_filtrado = df_filtrado[df_filtrado['Proveedor'].isin(proveedor)]
if solo_sospechosos:
    df_filtrado = df_filtrado[df_filtrado['Sospecha_Irregularidad']]

# Visualización general
st.subheader("Resumen de Contratos Analizados")
col1, col2, col3 = st.columns(3)
col1.metric("Contratos Totales", len(df))
col2.metric("Contratos Filtrados", len(df_filtrado))
col3.metric("Sospechas Detectadas", df['Sospecha_Irregularidad'].sum())

# Indicadores visuales
st.subheader("Indicadores de Irregularidad")

grafico_indicadores = df_filtrado[['Sin_Justificacion_CD', 'Pocas_Ofertas_Alto_Monto',
                                   'Plazo_Corto_Obras', 'Proveedor_Frecuente',
                                   'Adjudicacion_Rapida']].sum().reset_index()
grafico_indicadores.columns = ['Indicador', 'Cantidad']
fig = px.bar(grafico_indicadores, x='Indicador', y='Cantidad',
             title='Número de Contratos por Tipo de Indicador',
             text='Cantidad', color='Indicador')
st.plotly_chart(fig, use_container_width=True)

# Tabla de resultados
st.subheader("Tabla de Contratos con Indicadores")
st.dataframe(df_filtrado[['ID_Contrato', 'Organismo_Contratante', 'Proveedor',
                          'Tipo_Contratacion', 'Monto_Adjudicado', 'Sospecha_Irregularidad']])

# Exportar datos
st.download_button("📥 Descargar CSV de resultados", df_filtrado.to_csv(index=False), "resultados_filtrados.csv")