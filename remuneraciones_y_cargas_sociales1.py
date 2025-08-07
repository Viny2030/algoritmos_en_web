# =================================================================
# SCRIPT DE AUDITORÍA DE REMUNERACIONES CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# =================================================================

st.set_page_config (page_title="Auditoría de Remuneraciones y Cargas Sociales", layout="wide")


@st.cache_data
def generar_datos_nomina(num_empleados=100, fecha_inicio='2023-01-01', semilla=123):
    """Genera datos simulados de nómina."""
    if semilla is not None:
        np.random.seed (semilla)
        random.seed (semilla)
        Faker.seed (semilla)

    fake = Faker ('es_AR')
    departamentos = ['Ventas', 'Marketing', 'Finanzas', 'Operaciones', 'IT', 'Recursos Humanos']
    cargos = {
        'Ventas': ['Representante', 'Gerente'],
        'Marketing': ['Analista', 'Especialista'],
        'Finanzas': ['Contador', 'Analista Financiero'],
        'Operaciones': ['Coordinador', 'Supervisor'],
        'IT': ['Desarrollador', 'Ingeniero de Soporte'],
        'Recursos Humanos': ['Generalista', 'Reclutador']
    }

    fecha_inicio_dt = datetime.strptime (fecha_inicio, '%Y-%m-%d')
    datos = []

    for i in range (num_empleados):
        nombre = fake.first_name ()
        apellido = fake.last_name ()
        fecha_ingreso = fake.date_between (start_date=fecha_inicio_dt, end_date='today')
        departamento = random.choice (departamentos)
        cargo = random.choice (cargos[departamento])
        salario_bruto = round (random.uniform (50000, 300000), 2)

        aportes_patronales = salario_bruto * 0.23
        contribuciones_empleado = salario_bruto * 0.17
        salario_neto = salario_bruto - contribuciones_empleado

        datos.append ({
            'ID_Empleado': f'EMP-{i + 1:04d}',
            'Nombre': f'{nombre} {apellido}',
            'Departamento': departamento,
            'Cargo': cargo,
            'Fecha_Ingreso': fecha_ingreso,
            'Salario_Bruto': salario_bruto,
            'Aportes_Patronales': aportes_patronales,
            'Contribuciones_Empleado': contribuciones_empleado,
            'Salario_Neto': salario_neto
        })

    return pd.DataFrame (datos)


# =================================================================
# 3. LÓGICA DE ANÁLISIS
# =================================================================

def aplicar_auditoria(df):
    """Realiza el análisis de cargas sociales por departamento."""
    cargas_por_depto = df.groupby ('Departamento').agg (
        Salario_Total_Bruto=('Salario_Bruto', 'sum'),
        Aportes_Patronales_Total=('Aportes_Patronales', 'sum')
    ).reset_index ()
    cargas_por_depto['Carga_Social_Total'] = cargas_por_depto['Salario_Total_Bruto'] + cargas_por_depto[
        'Aportes_Patronales_Total']
    return cargas_por_depto


# =================================================================
# 4. INTERFAZ DE STREAMLIT
# =================================================================

st.title ("👨‍💼 Auditoría de Remuneraciones y Cargas Sociales")
st.markdown (
    "Esta aplicación audita datos simulados de nómina para analizar salarios, aportes y cargas sociales por departamento.")

if st.button ("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
    with st.spinner ('Ejecutando la auditoría...'):
        df_nomina = generar_datos_nomina (semilla=123)
        df_cargas_sociales = aplicar_auditoria (df_nomina)

        st.success ("✅ Auditoría completada con éxito.")

        # --- Sección 1: Resumen y Alertas ---
        st.header ("🔍 Resumen de Cargas Sociales por Departamento")

        st.dataframe (df_cargas_sociales.sort_values (by='Carga_Social_Total', ascending=False))

        csv_data = df_cargas_sociales.to_csv (index=False).encode ('utf-8')
        st.download_button (
            label="Descargar Resumen CSV",
            data=csv_data,
            file_name="resumen_cargas_sociales.csv",
            mime="text/csv"
        )

        # --- Sección 2: Visualizaciones ---
        st.header ("📈 Visualizaciones")

        # Gráfico 1: Salario Bruto por Departamento
        fig1, ax1 = plt.subplots (figsize=(12, 7))
        sns.boxplot (x='Departamento', y='Salario_Bruto', data=df_nomina, palette='viridis', ax=ax1)
        ax1.set_title ('1. Salario Bruto Promedio y Distribución por Departamento')
        ax1.set_xlabel ('Departamento')
        ax1.set_ylabel ('Salario Bruto')
        plt.xticks (rotation=45, ha='right')
        st.pyplot (fig1)

        # Gráfico 2: Carga Social Total por Departamento
        fig2, ax2 = plt.subplots (figsize=(12, 7))
        sns.barplot (x='Departamento', y='Carga_Social_Total', data=df_cargas_sociales, palette='magma', ax=ax2)
        ax2.set_title ('2. Carga Social Total (Salario Bruto + Aportes Patronales) por Departamento')
        ax2.set_xlabel ('Departamento')
        ax2.set_ylabel ('Carga Social Total')
        plt.xticks (rotation=45, ha='right')
        st.pyplot (fig2)

        # Gráfico 3: Distribución del Salario Neto
        fig3, ax3 = plt.subplots (figsize=(10, 6))
        sns.histplot (df_nomina['Salario_Neto'], bins=20, kde=True, color='purple', ax=ax3)
        ax3.set_title ('3. Distribución del Salario Neto')
        ax3.set_xlabel ('Salario Neto')
        ax3.set_ylabel ('Frecuencia')
        st.pyplot (fig3)