# ===============================================================
# SCRIPT DE CONCILIACIÓN BANCARIA CON STREAMLIT Y DOCKER
# ===============================================================

# --- 1. IMPORTACIONES ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz, process
import os
import streamlit as st

# ===============================================================
# 2. CONFIGURACIÓN Y GENERACIÓN DE DATOS SIMULADOS
# ===============================================================

# Configuración de la página de Streamlit
st.set_page_config (page_title="Conciliación Bancaria Automática", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos de movimientos bancarios y contables simulados."""
    fake = Faker ('es_AR')
    np.random.seed (123)
    random.seed (123)
    num_transacciones = 150
    tipos_transaccion = ['Débito', 'Crédito']
    conceptos_banco = ['Transferencia recibida', 'Transferencia enviada', 'Pago proveedor', 'Cobro cliente',
                       'Intereses', 'Comisión bancaria', 'Depósito efectivo']
    conceptos_contables = ['Pago proveedor', 'Cobro cliente', 'Intereses ganados', 'Comisión pagada',
                           'Depósito registrado']

    def generar_fecha():
        return fake.date_between (start_date='-60d', end_date='today')

    movimientos_banco = []
    movimientos_contables = []

    for i in range (num_transacciones):
        fecha = generar_fecha ()
        tipo = random.choice (tipos_transaccion)
        monto = round (random.uniform (500, 10000), 2)
        concepto_banco = random.choice (conceptos_banco)
        concepto_contable = random.choice (conceptos_contables)

        movimientos_banco.append ({
            'id_banco': i + 1, 'fecha': fecha, 'tipo': tipo,
            'concepto_banco': concepto_banco, 'monto_banco': monto,
            'referencia_banco': fake.unique.bothify ('BNK-########')
        })

        if random.random () > 0.1:
            monto_contable = monto if random.random () > 0.2 else round (monto + random.uniform (-100, 100), 2)
            movimientos_contables.append ({
                'id_contable': i + 1, 'fecha': fecha, 'tipo': tipo,
                'concepto_contable': concepto_contable, 'monto_contable': monto_contable,
                'referencia_contable': fake.unique.bothify ('CNT-########')
            })

    df_banco = pd.DataFrame (movimientos_banco)
    df_contabilidad = pd.DataFrame (movimientos_contables)

    return df_banco, df_contabilidad


# ===============================================================
# 3. PROCESO DE CONCILIACIÓN (Función principal)
# ===============================================================

def procesar_conciliacion(df_banco, df_contabilidad):
    """
    Función que encapsula todo el proceso de conciliación para
    ser llamada por la app de Streamlit.
    """
    # 3. Merge inicial
    df = pd.merge (df_banco, df_contabilidad, on=['fecha', 'tipo'], how='outer')

    # 4. Conciliación y clasificación inicial
    tolerancia = 50
    df['diferencia_monto'] = np.abs (df['monto_banco'] - df['monto_contable'])
    df['conciliado'] = df['diferencia_monto'] <= tolerancia
    df['conciliado'] = df['conciliado'].fillna (False)

    # Limpieza de datos (por si acaso)
    df['fecha'] = pd.to_datetime (df['fecha'])
    df['monto_banco'] = pd.to_numeric (df['monto_banco'], errors='coerce').fillna (0)
    df['monto_contable'] = pd.to_numeric (df['monto_contable'], errors='coerce').fillna (0)
    df['diferencia_monto'] = pd.to_numeric (df['diferencia_monto'], errors='coerce').fillna (0)

    conditions = [
        (df['conciliado'] == True) & (df['diferencia_monto'].abs () < 0.01),
        (df['conciliado'] == True) & (df['diferencia_monto'].abs () >= 0.01),
        (df['id_banco'].notna ()) & (df['id_contable'].isna ()),
        (df['id_banco'].isna ()) & (df['id_contable'].notna ()),
        (df['conciliado'] == False) & (df['id_banco'].notna ()) & (df['id_contable'].notna ())
    ]
    choices = ["Conciliado Exacto", "Conciliado con Diferencia", "Sólo Banco", "Sólo Contable",
               "No Conciliado - Diferencia Persistente"]
    df['clasificación_auditoría'] = np.select (conditions, choices, default='No Clasificado')

    audit_df = df.copy ()
    audit_df['antiguedad_dias'] = (pd.to_datetime (datetime.today ().date ()) - audit_df['fecha']).dt.days

    # 5. Fuzzy Matching
    solo_banco_df = audit_df[audit_df['clasificación_auditoría'] == 'Sólo Banco'].copy ()
    solo_contable_df = audit_df[audit_df['clasificación_auditoría'] == 'Sólo Contable'].copy ()
    monto_tolerance = 5.0
    fuzzy_threshold = 80
    date_tolerance_days = 7
    potential_matches = []
    matched_contable_indices = set ()

    for idx_b, row_b in solo_banco_df.iterrows ():
        monto_lower_bound = row_b['monto_banco'] - monto_tolerance
        monto_upper_bound = row_b['monto_banco'] + monto_tolerance
        potential_contable_matches = solo_contable_df[
            (solo_contable_df['fecha'].between (row_b['fecha'] - timedelta (days=date_tolerance_days),
                                                row_b['fecha'] + timedelta (days=date_tolerance_days)))
            & (solo_contable_df['monto_contable'].between (monto_lower_bound, monto_upper_bound))
            & (~solo_contable_df.index.isin (matched_contable_indices))
            ]

        if not potential_contable_matches.empty:
            bank_combined_string = f"{row_b['concepto_banco'] or ''} {row_b['referencia_banco'] or ''}".strip ()
            choices = {f"{(row_c['concepto_contable'] or '')} {(row_c['referencia_contable'] or '')}".strip (): idx_c
                       for idx_c, row_c in potential_contable_matches.iterrows ()}

            if bank_combined_string and choices:
                best_match = process.extractOne (bank_combined_string, choices.keys (), scorer=fuzz.token_sort_ratio)

                if best_match and best_match[1] >= fuzzy_threshold:
                    best_match_idx = choices[best_match[0]]
                    row_c = solo_contable_df.loc[best_match_idx]
                    potential_matches.append ({
                        'id_banco': row_b['id_banco'], 'fecha_banco': row_b['fecha'],
                        'monto_banco': row_b['monto_banco'],
                        'concepto_banco': row_b['concepto_banco'], 'referencia_banco': row_b['referencia_banco'],
                        'id_contable': row_c['id_contable'], 'fecha_contable': row_c['fecha'],
                        'monto_contable': row_c['monto_contable'],
                        'concepto_contable': row_c['concepto_contable'],
                        'referencia_contable': row_c['referencia_contable'],
                        'diferencia_monto_fuzzy': round (row_b['monto_banco'] - row_c['monto_contable'], 2),
                        'fuzzy_score': best_match[1]
                    })
                    audit_df.loc[idx_b, 'clasificación_auditoría'] = 'Potencialmente Conciliado (Fuzzy)'
                    audit_df.loc[best_match_idx, 'clasificación_auditoría'] = 'Potencialmente Conciliado (Fuzzy)'
                    matched_contable_indices.add (best_match_idx)

    fuzzy_matches_df = pd.DataFrame (potential_matches)
    solo_banco_restante = audit_df[audit_df['clasificación_auditoría'] == 'Sólo Banco'].copy ()

    return audit_df, fuzzy_matches_df, solo_banco_restante


# ===============================================================
# 4. INTERFAZ DE STREAMLIT
# ===============================================================

# Título y descripción
st.title ("📊 Conciliador Bancario Automático con Python")
st.markdown (
    "Esta aplicación simula un proceso de conciliación bancaria, comparando movimientos de la cuenta bancaria con los registros contables. Utiliza **Fuzzy Matching** para encontrar transacciones similares.")

# Botón para iniciar el proceso
if st.button ("Iniciar Proceso de Conciliación", help="Haz clic para generar datos simulados y correr el análisis"):
    with st.spinner ('Procesando la conciliación...'):
        df_banco, df_contabilidad = generar_datos_simulados ()
        audit_df, fuzzy_matches_df, solo_banco_restante = procesar_conciliacion (df_banco, df_contabilidad)

        st.success ("🎉 ¡Proceso completado con éxito!")

        # 4.1 Resumen y Métricas
        st.header ("🔍 Resumen de la Conciliación")
        col1, col2, col3 = st.columns (3)
        with col1:
            st.metric ("Total Movimientos Banco", len (df_banco))
        with col2:
            st.metric ("Total Movimientos Contables", len (df_contabilidad))
        with col3:
            st.metric ("Total de Movimientos Conciliados",
                       len (audit_df[audit_df['clasificación_auditoría'].str.contains ("Conciliado")]))

        st.subheader ("Clasificación de Transacciones")
        st.dataframe (audit_df['clasificación_auditoría'].value_counts ())

        # 4.2 Visualizaciones
        st.header ("📈 Visualizaciones Clave")

        # Gráfico 1: Conteo de transacciones
        fig1, ax1 = plt.subplots (figsize=(10, 6))
        sns.countplot (data=audit_df, y='clasificación_auditoría', palette='viridis',
                       order=audit_df['clasificación_auditoría'].value_counts ().index, ax=ax1)
        ax1.set_title ("1. Conteo de Transacciones por Tipo de Conciliación")
        ax1.set_xlabel ("Número de Transacciones")
        ax1.set_ylabel ("Clasificación de Auditoría")
        st.pyplot (fig1)

        # Gráfico 2: Distribución de diferencias
        fig2, ax2 = plt.subplots (figsize=(10, 6))
        sns.histplot (audit_df[audit_df['diferencia_monto'] > 0]['diferencia_monto'], kde=True, bins=30, color='coral',
                      ax=ax2)
        ax2.set_title ("2. Distribución de Diferencias Monetarias")
        ax2.set_xlabel ("Diferencia Absoluta ($)")
        ax2.set_ylabel ("Frecuencia")
        st.pyplot (fig2)

        # Gráfico 3: Antigüedad de transacciones no conciliadas
        no_conciliadas_df = audit_df[audit_df['clasificación_auditoría'].isin (
            ['Sólo Banco', 'Sólo Contable', 'No Conciliado - Diferencia Persistente'])]
        if not no_conciliadas_df.empty:
            fig3, ax3 = plt.subplots (figsize=(10, 6))
            sns.histplot (no_conciliadas_df['antiguedad_dias'], kde=True, bins=15, color='purple', ax=ax3)
            ax3.set_title ("3. Distribución de Antigüedad de Transacciones Pendientes")
            ax3.set_xlabel ("Antigüedad (Días)")
            ax3.set_ylabel ("Número de Transacciones")
            st.pyplot (fig3)
        else:
            st.info ("No hay transacciones pendientes para mostrar la antigüedad.")

        # 4.3 Tablas de resultados
        st.header ("📋 Tablas de Detalle")

        st.subheader ("Potenciales Conciliaciones (Fuzzy Matching)")
        if not fuzzy_matches_df.empty:
            st.dataframe (fuzzy_matches_df)
            st.download_button (
                label="Descargar Reporte Fuzzy CSV",
                data=fuzzy_matches_df.to_csv (index=False).encode ('utf-8'),
                file_name='reporte_conciliaciones_fuzzy.csv',
                mime='text/csv'
            )
        else:
            st.info ("No se encontraron potenciales conciliaciones por Fuzzy Matching.")

        st.subheader ("Transacciones Pendientes de Banco")
        if not solo_banco_restante.empty:
            st.dataframe (
                solo_banco_restante[['fecha', 'monto_banco', 'concepto_banco', 'referencia_banco', 'antiguedad_dias']])
            st.download_button (
                label="Descargar Reporte Pendientes CSV",
                data=solo_banco_restante.to_csv (index=False).encode ('utf-8'),
                file_name='reporte_solo_banco_restante.csv',
                mime='text/csv'
            )
        else:
            st.info ("No hay transacciones pendientes en el banco.")