# =================================================================
# SCRIPT DE AUDITORÍA DE ANTICIPOS DE CLIENTES CON STREAMLIT Y DOCKER
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import streamlit as st


# =================================================================
# 2. GENERACIÓN Y ANÁLISIS DE DATOS
# =================================================================

@st.cache_data
def generar_datos():
    """Genera datos sintéticos de anticipos de clientes."""
    np.random.seed(42)
    random.seed(42)
    fake = Faker('es_AR')
    Faker.seed(42)

    num_anticipos = 50
    fecha_actual_referencia = datetime.now()
    clientes = [f'Cliente_{i}' for i in range(1, 31)]
    monedas = ['USD', 'ARS', 'EUR']
    servicios_productos = [
        'Desarrollo de Software', 'Consultoría IT', 'Servicio de Mantenimiento', 'Licencia Anual',
        'Capacitación', 'Diseño Web', 'Soporte Técnico Premium', 'Implementación de ERP'
    ]
    estados_aplicacion = ['Pendiente de aplicación', 'Parcialmente aplicado', 'Totalmente aplicado']

    data = []
    for i in range(num_anticipos):
        id_cliente = random.choice(clientes)
        fecha_ant = fecha_actual_referencia - timedelta(days=random.randint(10, 365 * 2))
        monto = round(random.uniform(500, 25000), 2)
        servicio = random.choice(servicios_productos)
        estado_aplic = random.choices(estados_aplicacion, weights=[0.5, 0.3, 0.2], k=1)[0]
        fecha_cierre = pd.NaT

        if estado_aplic == 'Totalmente aplicado':
            monto_aplicado = monto
            fecha_cierre = fecha_ant + timedelta(days=random.randint(1, 90))
            if fecha_cierre > fecha_actual_referencia:
                fecha_cierre = fecha_actual_referencia - timedelta(days=random.randint(1, 30))
            if fecha_cierre < fecha_ant:
                fecha_cierre = fecha_ant + timedelta(days=1)
        elif estado_aplic == 'Parcialmente aplicado':
            monto_aplicado = round(random.uniform(0.1, 0.9) * monto, 2)
        else:
            monto_aplicado = 0.0

        data.append({
            'id_anticipo': f'ANT-{i:04d}', 'id_cliente': id_cliente, 'fecha_anticipo': fecha_ant,
            'monto_anticipo': monto, 'moneda': random.choice(monedas),
            'servicio_producto_asociado': servicio, 'estado_aplicacion': estado_aplic,
            'monto_aplicado': monto_aplicado,
            'monto_pendiente_aplicar': round(monto - monto_aplicado, 2),
            'fecha_cierre_aplicacion': fecha_cierre
        })

    return pd.DataFrame(data), fecha_actual_referencia


@st.cache_data
def analizar_datos(df_anticipos, fecha_actual_referencia):
    """Aplica el análisis de auditoría y detección de anomalías."""
    df_anticipos['fecha_anticipo'] = pd.to_datetime(df_anticipos['fecha_anticipo'])
    df_anticipos['fecha_cierre_aplicacion'] = pd.to_datetime(df_anticipos['fecha_cierre_aplicacion'])
    numeric_cols = ['monto_anticipo', 'monto_aplicado', 'monto_pendiente_aplicar']
    for col in numeric_cols:
        df_anticipos[col] = pd.to_numeric(df_anticipos[col], errors='coerce').fillna(0)

    df_anticipos['dias_desde_anticipo'] = (fecha_actual_referencia - df_anticipos['fecha_anticipo']).dt.days

    df_pendientes = df_anticipos[df_anticipos['monto_pendiente_aplicar'] > 0].copy()
    if not df_pendientes.empty:
        bins_antiguedad = [-np.inf, 30, 90, 180, 365, np.inf]
        labels_antiguedad = ['0-30 days', '31-90 days', '91-180 days', '181-365 days', '> 365 days']
        df_pendientes['rango_antiguedad_pendiente'] = pd.cut(
            df_pendientes['dias_desde_anticipo'],
            bins=bins_antiguedad, labels=labels_antiguedad, right=False
        )

    df_anticipos['monto_anticipo_zscore'] = zscore(df_anticipos['monto_anticipo'])
    umbral_zscore = 2.5
    anomalias_monto_anticipo = df_anticipos[df_anticipos['monto_anticipo_zscore'].abs() > umbral_zscore]

    df_active_anticipos = df_anticipos[df_anticipos['estado_aplicacion'] != 'Totalmente aplicado'].copy()
    if not df_active_anticipos.empty:
        features_ia = df_active_anticipos[['monto_anticipo', 'dias_desde_anticipo']].fillna(0)
        iso_forest = IsolationForest(random_state=42, contamination=0.1)
        iso_forest.fit(features_ia)
        df_active_anticipos['is_anomaly_ia'] = iso_forest.predict(features_ia)
        df_anticipos = df_anticipos.merge(df_active_anticipos[['id_anticipo', 'is_anomaly_ia']], on='id_anticipo',
                                           how='left')
        df_anticipos['is_anomaly_ia'].fillna(1, inplace=True)
    else:
        df_anticipos['is_anomaly_ia'] = 1

    return df_anticipos, df_pendientes, anomalias_monto_anticipo, umbral_zscore


# =================================================================
# 3. INTERFAZ DE STREAMLIT
# =================================================================
def main():
    st.set_page_config(layout="wide", page_title="Análisis de Anticipos de Clientes")
    st.title('💰 Auditoría y Análisis de Anticipos de Clientes')
    st.markdown("""
        Esta aplicación genera datos sintéticos de anticipos de clientes y realiza un análisis
        completo para auditoría y detección de anomalías.
    """)

    if st.button("Iniciar Auditoría", help="Genera datos simulados y aplica el análisis completo"):
        with st.spinner('Ejecutando el análisis, por favor espere...'):
            df_anticipos, fecha_actual_referencia = generar_datos()
            df_auditado, df_pendientes, anomalias_monto_anticipo, umbral_zscore = analizar_datos(df_anticipos.copy(),
                                                                                                  fecha_actual_referencia)

        st.success("✅ Auditoría completada con éxito.")

        # Resumen de métricas
        st.subheader('📝 Resumen de Auditoría')

        col1, col2, col3 = st.columns(3)

        total_anticipos = len(df_auditado)
        monto_total_anticipos = df_auditado['monto_anticipo'].sum()
        monto_pendiente = df_auditado['monto_pendiente_aplicar'].sum()

        col1.metric("Total de Anticipos", total_anticipos)
        col2.metric("Monto Total Anticipado", f"${monto_total_anticipos:,.2f}")
        col3.metric("Monto Pendiente de Aplicar", f"${monto_pendiente:,.2f}")

        anomalias_ia_count = (df_auditado['is_anomaly_ia'] == -1).sum()
        anomalias_zscore_count = len(anomalias_monto_anticipo)

        st.markdown("---")
        st.markdown(f"**Anomalías detectadas:**")
        st.markdown(f"- Anomalías por monto (Z-score > {umbral_zscore}): **{anomalias_zscore_count}**")
        st.markdown(f"- Anomalías por Isolation Forest (en anticipos activos): **{anomalias_ia_count}**")

        st.subheader('🔍 Datos Generados (Vista Previa)')
        st.dataframe(df_auditado.head())

        # =================================================================
        # PARTE 4: VISUALIZACIÓN DE RESULTADOS (GRÁFICOS)
        # =================================================================
        st.subheader('📈 Visualización de Resultados')
        sns.set(style="whitegrid")

        # Graph 1 & 2: Montos por Cliente
        st.write('### 1. Clientes por Monto de Anticipos')
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            monto_por_cliente = df_auditado.groupby('id_cliente')['monto_anticipo'].sum().sort_values(
                ascending=False)
            sns.barplot(x=monto_por_cliente.head(10).index, y=monto_por_cliente.head(10).values, palette='viridis',
                         ax=ax1)
            ax1.set_title('Top 10 Clientes por Monto Total')
            # Corrección: Se eliminó ha='right'
            ax1.tick_params(axis='x', rotation=45)
            st.pyplot(fig1)

        with col2:
            if not df_pendientes.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                monto_pendiente_por_cliente = df_auditado.groupby('id_cliente')[
                    'monto_pendiente_aplicar'].sum().sort_values(ascending=False)
                top_pendientes = monto_pendiente_por_cliente[monto_pendiente_por_cliente > 0].head(10)
                sns.barplot(x=top_pendientes.index, y=top_pendientes.values, palette='magma', ax=ax2)
                ax2.set_title('Top 10 Clientes por Monto Pendiente')
                # Corrección: Se eliminó ha='right'
                ax2.tick_params(axis='x', rotation=45)
                st.pyplot(fig2)

        # Graph 3 & 4: Distribución por Estado
        st.write('### 2. Distribución de Anticipos por Estado de Aplicación')
        col3, col4 = st.columns(2)
        with col3:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.countplot(x='estado_aplicacion', data=df_auditado, palette='cividis',
                           order=df_auditado['estado_aplicacion'].value_counts().index, ax=ax3)
            ax3.set_title('Distribución por Estado')
            st.pyplot(fig3)

        with col4:
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            monto_total_por_estado_aplicacion = df_auditado.groupby('estado_aplicacion')[
                'monto_anticipo'].sum().sort_values(ascending=False)
            sns.barplot(x=monto_total_por_estado_aplicacion.index, y=monto_total_por_estado_aplicacion.values,
                         palette='plasma', ax=ax4)
            ax4.set_title('Monto Total por Estado')
            st.pyplot(fig4)

        # Graph 5 & 6: Detección de Anomalías
        st.write('### 3. Detección de Anomalías')
        col5, col6 = st.columns(2)
        with col5:
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=df_auditado.index, y='monto_anticipo', data=df_auditado, label='Anticipos', color='blue',
                             alpha=0.6, ax=ax5)
            if not anomalias_monto_anticipo.empty:
                sns.scatterplot(x=anomalias_monto_anticipo.index, y='monto_anticipo', data=anomalias_monto_anticipo,
                                 color='red', s=120, label=f'Anomalía (Z-score > {umbral_zscore})', marker='X', ax=ax5)
            ax5.set_title('Anomalías en Montos (Z-score)')
            ax5.legend()
            st.pyplot(fig5)

        with col6:
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='monto_anticipo', y='dias_desde_anticipo', data=df_auditado, hue='is_anomaly_ia',
                             style='is_anomaly_ia', palette={1: 'blue', -1: 'red'}, markers={1: 'o', -1: 'X'}, s=100,
                             ax=ax6)
            ax6.set_title('Anomalías (IA): Monto vs. Antigüedad')
            handles, labels = ax6.get_legend_handles_labels()
            ax6.legend(handles, ['Normal', 'Anomalía'], title='Resultado IA')
            st.pyplot(fig6)

        # Graph 7 & 8: Antigüedad y Moneda
        st.write('### 4. Distribución de Antigüedad y Moneda')
        col7, col8 = st.columns(2)
        with col7:
            if not df_pendientes.empty:
                fig7, ax7 = plt.subplots(figsize=(10, 6))
                monto_pendiente_por_rango = df_pendientes.groupby('rango_antiguedad_pendiente', observed=True)[
                    'monto_pendiente_aplicar'].sum()
                sns.barplot(x=monto_pendiente_por_rango.index, y=monto_pendiente_por_rango.values, palette='rocket',
                             ax=ax7)
                ax7.set_title('Monto Pendiente por Antigüedad')
                # Corrección: Se eliminó ha='right'
                ax7.tick_params(axis='x', rotation=45)
                st.pyplot(fig7)
        with col8:
            fig8, ax8 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='moneda', y='monto_anticipo', data=df_auditado, palette='pastel', ax=ax8)
            ax8.set_title('Distribución de Montos por Moneda')
            st.pyplot(fig8)


if __name__ == '__main__':
    main()
