# ===============================================================
# PARTE 1: GENERACIÓN DE DATOS (Función)
# ===============================================================
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st


@st.cache_data
def generar_dataframe_caja():
    """Genera y devuelve un DataFrame simulado de transacciones de caja."""
    fake_es = Faker('es_AR')
    np.random.seed(42)
    random.seed(42)
    Faker.seed(42)

    descripciones_gasto = [
        "Pago de servicios", "Compra de suministros", "Mantenimiento de equipos",
        "Gastos de representación", "Alquiler de oficina", "Pago a proveedores",
        "Reparación de maquinaria", "Transporte y logística", "Capacitación del personal",
        "Honorarios profesionales"
    ]
    num_registros = 50  # Aumentado para mejor visualización
    responsables = [fake_es.name() for _ in range(10)]
    tipos_transaccion = ['Venta', 'Gasto']
    metodos_pago = ['Efectivo', 'Tarjeta de Débito', 'Tarjeta de Crédito', 'Transferencia']
    categorias_productos = ['Electrónica', 'Alimentos', 'Ropa', 'Accesorios', 'Juguetes']
    saldo = 50000
    registros = []

    for i in range(num_registros):
        fecha_hora_transaccion = fake_es.date_time_between(start_date='-6M', end_date='now')
        tipo_transaccion = random.choice(tipos_transaccion)
        monto = round(random.uniform(1000, 15000), 2)

        if tipo_transaccion == 'Venta':
            saldo += monto
        else:
            saldo -= monto

        registro = {
            'id_transaccion': i + 1,
            'fecha_hora': fecha_hora_transaccion.strftime('%Y-%m-%d %H:%M:%S'),
            'tipo_transaccion': tipo_transaccion,
            'metodo_pago': random.choice(metodos_pago),
            'monto': monto,
            'saldo_acumulado': round(saldo, 2),
            'cajero_id': fake_es.random_int(min=1, max=10),
            'numero_ticket': fake_es.unique.bothify(text='TK-########'),
            'cliente_id': fake_es.random_int(min=1000, max=9999) if random.random() > 0.3 else None,
            'producto_categoria': random.choice(categorias_productos) if tipo_transaccion == 'Venta' else None,
            'descripcion': random.choice(descripciones_gasto) if tipo_transaccion == 'Gasto' else None,
            'responsable': random.choice(responsables)
        }
        registros.append(registro)

    df = pd.DataFrame(registros)
    df.sort_values(by='fecha_hora', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ===============================================================
# PARTE 2: ANÁLISIS DE AUDITORÍA Y DETECCIÓN DE ANOMALÍAS (Función)
# ===============================================================
@st.cache_data
def analizar_datos(df):
    """Aplica reglas de auditoría y detección de anomalías al DataFrame."""
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    df['diferencia_saldo'] = df['saldo_acumulado'].diff().fillna(0).round(2)
    df['error_saldo'] = df.apply(
        lambda row: 'Venta con saldo decreciente' if row['tipo_transaccion'] == 'Venta' and row[
            'diferencia_saldo'] < 0 else ('Gasto con saldo creciente' if row['tipo_transaccion'] == 'Gasto' and row[
            'diferencia_saldo'] > 0 else None), axis=1)
    df['error_descripcion'] = df.apply(
        lambda r: 'Gasto sin descripción' if r['tipo_transaccion'] == 'Gasto' and pd.isnull(
            r['descripcion']) else None, axis=1)
    df['error_categoria'] = df.apply(
        lambda r: 'Venta sin categoría' if r['tipo_transaccion'] == 'Venta' and pd.isnull(
            r['producto_categoria']) else None, axis=1)
    df['hora'] = df['fecha_hora'].dt.hour
    df['error_horario'] = df.apply(lambda r: 'Gasto fuera de horario (7-21)' if r['tipo_transaccion'] == 'Gasto' and (
                r['hora'] < 7 or r['hora'] > 21) else None, axis=1)
    gastos_por_cajero = df[df['tipo_transaccion'] == 'Gasto'].groupby('cajero_id').size()
    cajeros_sospechosos = gastos_por_cajero[gastos_por_cajero > 3].index.tolist()
    df['alerta_cajero'] = df.apply(lambda r: 'Cajero con muchos gastos (>3)' if r['tipo_transaccion'] == 'Gasto' and r[
        'cajero_id'] in cajeros_sospechosos else None, axis=1)
    df['alerta_duplicada'] = df.duplicated(['id_transaccion', 'fecha_hora'], keep=False).apply(
        lambda x: 'Transacción duplicada' if x else None)

    def detectar_outliers_iqr(df_segmento, columna_monto):
        if df_segmento.empty: return pd.Series([False] * len(df_segmento), index=df_segmento.index)
        Q1, Q3 = df_segmento[columna_monto].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        if IQR == 0: return pd.Series(False, index=df_segmento.index)
        limite_inferior, limite_superior = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return ~df_segmento[columna_monto].between(limite_inferior, limite_superior)

    for tipo in df['tipo_transaccion'].unique():
        subset_indices = df['tipo_transaccion'] == tipo
        is_outlier = detectar_outliers_iqr(df[subset_indices], 'monto')
        df.loc[subset_indices, 'alerta_monto_irregular'] = is_outlier.apply(
            lambda x: f'Monto {tipo} irregular (outlier)' if x else None)

    features_ia = df[['monto', 'diferencia_saldo']].copy()
    iso_forest = IsolationForest(random_state=42, contamination=0.05)
    df['is_anomaly'] = iso_forest.fit_predict(features_ia)
    df['alerta_fraude_ia'] = df['is_anomaly'].apply(lambda x: 'Anomalía detectada por IA' if x == -1 else None)
    df['dia_semana_es'] = df['fecha_hora'].dt.day_name().replace(
        {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
         'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    )
    alert_cols = ['error_saldo', 'error_descripcion', 'error_categoria', 'error_horario',
                  'alerta_cajero', 'alerta_duplicada', 'alerta_monto_irregular', 'alerta_fraude_ia']
    df['alertas'] = df[alert_cols].apply(lambda row: ', '.join(row.dropna()), axis=1)
    df_alertas = df[df['alertas'] != ''].copy()

    return df, df_alertas


# ===============================================================
# PARTE 3: APLICACIÓN STREAMLIT
# ===============================================================
def main():
    st.set_page_config(layout="wide", page_title="Auditoría de Caja")
    st.title('💰 Auditoría y Detección de Fraude en Caja')
    st.markdown("""
        Esta aplicación simula transacciones de caja y aplica reglas de auditoría y modelos
        de IA para detectar posibles anomalías y fraudes.
    """)

    df = generar_dataframe_caja()
    df, df_alertas = analizar_datos(df)

    st.subheader('🔍 Datos Generados (Vista Previa)')
    st.dataframe(df.head(10))

    st.subheader('📝 Resumen de Alertas')

    total_transacciones = len(df)
    total_alertas = len(df_alertas)
    porcentaje_alertas = (total_alertas / total_transacciones) * 100 if total_transacciones > 0 else 0
    total_anomalias_ia = (df['alerta_fraude_ia'] == 'Anomalía detectada por IA').sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Transacciones Analizadas", total_transacciones)
    col2.metric("Transacciones con Alertas", total_alertas, f"{porcentaje_alertas:.2f}%")
    col3.metric("Anomalías por IA", total_anomalias_ia)

    if not df_alertas.empty:
        st.write('**Transacciones con Alertas**')
        st.dataframe(df_alertas[['id_transaccion', 'fecha_hora', 'monto', 'alertas']])

    st.subheader('📈 Visualizaciones de la Auditoría')
    sns.set(style="whitegrid")

    # --- Gráfico 1: Evolución del Saldo ---
    st.write('### 1. Evolución del Saldo Acumulado')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='fecha_hora', y='saldo_acumulado', hue='tipo_transaccion', marker='o', ax=ax1)
    ax1.set_title('Evolución del Saldo Acumulado')
    ax1.set_ylabel('Saldo Acumulado ($)')
    ax1.set_xlabel('Fecha')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # --- Gráfico 2: Alertas por Día de la Semana ---
    if not df_alertas.empty:
        st.write('### 2. Cantidad de Alertas por Día de la Semana')
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        orden_dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        sns.countplot(data=df_alertas, x='dia_semana_es', order=orden_dias_es, palette='plasma', hue='dia_semana_es',
                       legend=False, ax=ax2)
        ax2.set_title('Cantidad de Alertas por Día de la Semana')
        ax2.set_xlabel('Día de la Semana')
        ax2.set_ylabel('Número de Alertas')
        st.pyplot(fig2)

    # --- Gráfico 3: Alertas por Hora del Día ---
    st.write('### 3. Distribución de Transacciones con Alertas por Hora del Día')
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    if not df_alertas.empty:
        sns.histplot(data=df_alertas, x='hora', bins=24, kde=True, ax=ax3)
        ax3.set_title('Distribución de Alertas por Hora del Día')
        ax3.set_xlabel('Hora del Día')
        ax3.set_ylabel('Número de Alertas')
    else:
        st.info("No hay alertas para mostrar la distribución por hora.")
    st.pyplot(fig3)


if __name__ == '__main__':
    main()