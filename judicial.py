import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página de Streamlit
st.set_page_config(page_title="Análisis de Casos Judiciales", layout="wide")

# Título de la aplicación
st.title("Análisis de Casos Judiciales")
st.markdown("Explora y visualiza datos de casos judiciales.")

# Carga de datos desde la URL
url = "https://raw.githubusercontent.com/Viny2030/Libro_Algoritmos_contra_fraude_corrupcion/refs/heads/main/df_casos_judiciales.csv"
try:
    data_casos = pd.read_csv(url)
    df_casos = pd.DataFrame(data_casos)
    st.success("Datos cargados exitosamente desde la URL.")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Sidebar para filtros
st.sidebar.header("Filtros")
tipo_causa_seleccionado = st.sidebar.multiselect("Filtrar por Tipo de Causa:", df_casos['Tipo_Causa'].unique())
fallo_seleccionado = st.sidebar.multiselect("Filtrar por Fallo:", df_casos['Fallo'].unique())
instancia_seleccionada = st.sidebar.multiselect("Filtrar por Instancia:", df_casos['Instancia'].unique())
estado_seleccionado = st.sidebar.multiselect("Filtrar por Estado Actual:", df_casos['Estado_Actual'].unique())

# Aplicar filtros al DataFrame
df_filtrado = df_casos.copy()
if tipo_causa_seleccionado:
    df_filtrado = df_filtrado[df_filtrado['Tipo_Causa'].isin(tipo_causa_seleccionado)]
if fallo_seleccionado:
    df_filtrado = df_filtrado[df_filtrado['Fallo'].isin(fallo_seleccionado)]
if instancia_seleccionada:
    df_filtrado = df_filtrado[df_filtrado['Instancia'].isin(instancia_seleccionada)]
if estado_seleccionado:
    df_filtrado = df_filtrado[df_filtrado['Estado_Actual'].isin(estado_seleccionado)]

st.subheader("Datos Filtrados")
st.dataframe(df_filtrado)

st.markdown("---")
st.subheader("Análisis de Datos")

# 1. Tiempo de Proceso
st.subheader("1. Análisis del Tiempo de Proceso")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tiempo Promedio de Proceso (días)", f"{df_filtrado['Tiempo_Proceso_Dias'].mean():.2f}")
with col2:
    st.metric("Tiempo Mínimo de Proceso (días)", df_filtrado['Tiempo_Proceso_Dias'].min())
with col3:
    st.metric("Tiempo Máximo de Proceso (días)", df_filtrado['Tiempo_Proceso_Dias'].max())

fig_tiempo_proceso, ax_tiempo_proceso = plt.subplots()
sns.histplot(df_filtrado['Tiempo_Proceso_Dias'], bins=20, kde=True, ax=ax_tiempo_proceso)
ax_tiempo_proceso.set_title('Distribución del Tiempo de Proceso (en días)')
ax_tiempo_proceso.set_xlabel('Tiempo de Proceso (Días)')
ax_tiempo_proceso.set_ylabel('Frecuencia')
st.pyplot(fig_tiempo_proceso)

st.markdown("---")

# 2. Análisis de Fallos
st.subheader("2. Análisis de Fallos")
fallos_counts = df_filtrado['Fallo'].value_counts()
fig_fallos, ax_fallos = plt.subplots()
ax_fallos.pie(fallos_counts, labels=fallos_counts.index, autopct='%1.1f%%', startangle=140)
ax_fallos.set_title('Distribución de los Fallos')
ax_fallos.axis('equal')
st.pyplot(fig_fallos)

st.markdown("---")

# 3. Tiempo de Proceso por Tipo de Causa
st.subheader("3. Tiempo de Proceso por Tipo de Causa")
tiempo_por_causa = df_filtrado.groupby('Tipo_Causa')['Tiempo_Proceso_Dias'].mean().sort_values(ascending=False)

fig_tiempo_causa, ax_tiempo_causa = plt.subplots()
tiempo_por_causa.plot(kind='bar', ax=ax_tiempo_causa)
ax_tiempo_causa.set_title('Tiempo Promedio de Proceso por Tipo de Causa')
ax_tiempo_causa.set_ylabel('Tiempo Promedio (Días)')
ax_tiempo_causa.set_xlabel('Tipo de Causa')
ax_tiempo_causa.set_xticklabels(ax_tiempo_causa.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig_tiempo_causa)

st.markdown("---")

# 4. Fallos por Tipo de Causa
st.subheader("4. Fallos por Tipo de Causa")
fallos_por_causa = df_filtrado.groupby(['Tipo_Causa', 'Fallo']).size().unstack(fill_value=0)
fig_fallos_causa, ax_fallos_causa = plt.subplots()
fallos_por_causa.plot(kind='bar', stacked=True, ax=ax_fallos_causa)
ax_fallos_causa.set_title('Distribución de Fallos por Tipo de Causa')
ax_fallos_causa.set_ylabel('Número de Casos')
ax_fallos_causa.set_xlabel('Tipo de Causa')

# ✅ Corrección aquí: rotar y alinear etiquetas del eje x
for label in ax_fallos_causa.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')

plt.tight_layout()
st.pyplot(fig_fallos_causa)

st.markdown("---")

# 5. Tiempo de Proceso por Estado Actual
st.subheader("5. Tiempo de Proceso por Estado Actual")
tiempo_por_estado = df_filtrado.groupby('Estado_Actual')['Tiempo_Proceso_Dias'].mean().sort_values(ascending=False)
fig_tiempo_estado, ax_tiempo_estado = plt.subplots()
tiempo_por_estado.plot(kind='bar', ax=ax_tiempo_estado)
ax_tiempo_estado.set_title('Tiempo Promedio de Proceso por Estado Actual')
ax_tiempo_estado.set_ylabel('Tiempo Promedio (Días)')
ax_tiempo_estado.set_xlabel('Estado Actual')

# ✅ Corrección aquí también
for label in ax_tiempo_estado.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')

plt.tight_layout()
st.pyplot(fig_tiempo_estado)

st.markdown("---")

# 6. Fallos por Instancia
st.subheader("6. Fallos por Instancia")
fallos_por_instancia = df_filtrado.groupby(['Instancia', 'Fallo']).size().unstack(fill_value=0)
fig_fallos_instancia, ax_fallos_instancia = plt.subplots()
fallos_por_instancia.plot(kind='bar', stacked=True, ax=ax_fallos_instancia)
ax_fallos_instancia.set_title('Distribución de Fallos por Instancia')
ax_fallos_instancia.set_ylabel('Número de Casos')
ax_fallos_instancia.set_xlabel('Instancia')

# ✅ Corrección aquí también
for label in ax_fallos_instancia.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')

plt.tight_layout()
st.pyplot(fig_fallos_instancia)

st.markdown("---")
st.info("Esta aplicación fue desarrollada con Streamlit para el análisis de datos judiciales.")
