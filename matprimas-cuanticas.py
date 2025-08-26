# =================================================================
# SCRIPT DE AUDITORÍA CUÁNTICA DE MATERIAS PRIMAS CON STREAMLIT Y QISKIT
# =================================================================
import streamlit as st
# --- 1. IMPORTACIONES CUÁNTICAS Y CLÁSICAS UNIFICADAS ---


# =================================================================
# 2. CONFIGURACIÓN DE PÁGINA Y GENERACIÓN DE DATOS
# =================================================================

st.set_page_config (page_title="Auditoría Cuántica de Materias Primas", layout="wide")


@st.cache_data
def generar_datos_simulados():
    """Genera datos simulados de auditoría para la demostración."""
    np.random.seed (42)
    num_registros = 100
    fechas = pd.date_range (end=datetime.now (), periods=num_registros, freq='D')
    proveedores = [f'Proveedor_{i}' for i in random.choices (range (1, 15), k=num_registros)]
    cantidades = [random.randint (5, 500) for _ in range (num_registros)]
    precios = [round (random.uniform (50, 1000), 2) for _ in range (num_registros)]
    calidades = random.choices (['A', 'B', 'C'], weights=[0.6, 0.3, 0.1], k=num_registros)

    # Añadir algunas anomalías
    cantidades[-5:] = [1000, 1200, 1500, 2000, 100]  # Cantidades muy altas o muy bajas
    precios[-5:] = [1500, 1800, 20, 2000, 5]  # Precios fuera de rango

    df = pd.DataFrame ({
        'fecha': fechas,
        'proveedor': proveedores,
        'cantidad': cantidades,
        'precio_unitario': precios,
        'calidad': calidades
    })
    df['costo_total'] = df['cantidad'] * df['precio_unitario']
    return df


# =================================================================
# 3. LÓGICA DE AUDITORÍA CLÁSICA
# =================================================================

def aplicar_auditoria_clasica(df):
    """
    Aplica algoritmos clásicos de detección de anomalías (Isolation Forest).
    """
    st.subheader ("🤖 Detección de Anomalías (Clásica - Isolation Forest)")

    features = df[['costo_total', 'cantidad']].copy ()
    features['costo_total'] = StandardScaler ().fit_transform (features[['costo_total']])
    features['cantidad'] = StandardScaler ().fit_transform (features[['cantidad']])

    iso_forest = IsolationForest (contamination=0.1, random_state=42)
    df['anomalia_clasica'] = iso_forest.fit_predict (features)

    anomalias = df[df['anomalia_clasica'] == -1]

    if not anomalias.empty:
        st.warning (f"Se detectaron {len (anomalias)} anomalías usando Isolation Forest.")
        st.dataframe (anomalias)
    else:
        st.info ("No se detectaron anomalías con el modelo clásico.")

    fig, ax = plt.subplots ()
    sns.scatterplot (x='costo_total', y='cantidad', hue='anomalia_clasica',
                     palette={1: 'blue', -1: 'red'}, data=df, ax=ax)
    ax.set_title ("Anomalías Clásicas (Isolation Forest)")
    st.pyplot (fig)

    return df


# =================================================================
# 4. LÓGICA DE AUDITORÍA CUÁNTICA
# =================================================================

def aplicar_auditoria_cuantica(df):
    """
    Aplica un clasificador cuántico para la detección de anomalías.
    """
    st.subheader ("⚛️ Detección de Anomalías (Cuántica)")

    # 1. Preparación de datos para el modelo cuántico
    df_for_qml = df.copy ()

    # Para la demostración, las anomalías clásicas se usan como etiquetas para el modelo cuántico
    X = df_for_qml[['costo_total', 'cantidad']].values
    y = np.where (df_for_qml['anomalia_clasica'] == -1, 0, 1)  # 0 para anomalía, 1 para normal

    scaler = StandardScaler ()
    X_scaled = scaler.fit_transform (X)

    # 2. Definir los componentes cuánticos
    num_features = X_scaled.shape[1]

    # Mapeo de características (Quantum Feature Map)
    feature_map = ZZFeatureMap (feature_dimension=num_features, reps=2, entanglement='linear')

    # Circuito Análogo a una Red Neuronal (Quantum Variational Circuit)
    ansatz = RealAmplitudes (num_qubits=num_features, reps=1)

    # 3. Construir la Red Neuronal Cuántica (QNN)
    qnn = EstimatorQNN (
        circuit=ansatz,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )

    # 4. Configurar el clasificador y optimizador
    optimizer = COBYLA (maxiter=100)
    classifier = NeuralNetworkClassifier (qnn, optimizer=optimizer)

    # 5. Entrenar el clasificador
    with st.spinner ('Entrenando el clasificador cuántico...'):
        classifier.fit (X_scaled, y)

    # 6. Predecir y evaluar
    y_pred = classifier.predict (X_scaled)
    df['anomalia_cuantica'] = np.where (y_pred == 0, -1, 1)

    anomalias_qml = df[df['anomalia_cuantica'] == -1]

    if not anomalias_qml.empty:
        st.warning (f"Se detectaron {len (anomalias_qml)} anomalías con el clasificador cuántico.")
        st.dataframe (anomalias_qml)
    else:
        st.info ("No se detectaron anomalías con el modelo cuántico.")

    fig, ax = plt.subplots ()
    sns.scatterplot (x='costo_total', y='cantidad', hue='anomalia_cuantica',
                     palette={1: 'blue', -1: 'red'}, data=df, ax=ax)
    ax.set_title ("Anomalías Cuánticas")
    st.pyplot (fig)

    return df


# =================================================================
# 5. LÓGICA PRINCIPAL DE STREAMLIT
# =================================================================

st.title ("📦 Auditoría de Materias Primas")
st.markdown ("Esta aplicación audita datos simulados de materias primas con algoritmos clásicos y cuánticos.")

if st.button ("Iniciar Auditoría Completa"):
    df_materias_primas = generar_datos_simulados ()
    st.dataframe (df_materias_primas.head ())

    # Sección Clásica
    df_auditado_clasico = aplicar_auditoria_clasica (df_materias_primas.copy ())

    # Sección Cuántica
    aplicar_auditoria_cuantica (df_auditado_clasico.copy ())