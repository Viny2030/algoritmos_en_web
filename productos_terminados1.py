# =================================================================
# SCRIPT DE AUDITORÍA DE PRODUCTOS TERMINADOS (VERSIÓN DOCKER)
# =================================================================

# --- 1. IMPORTACIONES UNIFICADAS ---
import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os # <-- MODIFICACIÓN 1: Importar OS

print("✅ Librerías cargadas correctamente.")

# --- MODIFICACIÓN 2: Crear una carpeta de salida para todos los resultados ---
output_dir = 'resultados'
os.makedirs(output_dir, exist_ok=True)


# --- 2. GENERACIÓN DE DATOS SIMULADOS ---
print("--- Iniciando la generación de datos sintéticos... ---")
np.random.seed(42)
random.seed(42)
fake = Faker('es_AR')
Faker.seed(42)

num_registros_wip = 50
productos_base = ['Silla Estándar', 'Mesa de Comedor', 'Estantería Modular', 'Armario Doble', 'Puerta Interior']
etapas_produccion = ['Corte', 'Ensamblado', 'Soldadura', 'Pintura', 'Control de Calidad']
lineas_produccion = ['Línea A', 'Línea B', 'Línea C']

productos_en_proceso_temp = []
for i in range(num_registros_wip):
    producto = random.choice(productos_base)
    lote = f'L-{fake.random_int(min=1000, max=9999)}'
    cantidad_total = random.randint(50, 200)
    fecha_inicio = fake.date_between(start_date='-60d', end_date='-7d')
    duracion_estimada = random.randint(3, 20)
    fecha_estim_termino = fecha_inicio + timedelta(days=duracion_estimada)
    avance_porcentaje = random.randint(70, 100)
    cantidad_en_proceso_calculada = int(cantidad_total * ((100 - avance_porcentaje) / 100))
    cantidad_terminada_calculada = cantidad_total - cantidad_en_proceso_calculada
    productos_en_proceso_temp.append({
        'id_proceso': f'WIP-{1000 + i}', 'producto': producto, 'lote': lote,
        'cantidad_terminada_wip': cantidad_terminada_calculada,
        'fecha_estim_termino_proceso': fecha_estim_termino, 'linea_produccion': random.choice(lineas_produccion),
        'avance_porcentaje_wip': avance_porcentaje
    })
df_wip_temp = pd.DataFrame(productos_en_proceso_temp)

productos_terminados_final = []
for _, row in df_wip_temp.iterrows():
    if row['cantidad_terminada_wip'] > 0:
        fecha_termino_real = row['fecha_estim_termino_proceso'] + timedelta(days=random.randint(0, 5))
        estado_pt = 'En Stock' if row['avance_porcentaje_wip'] == 100 else 'Parcialmente en Almacén'
        productos_terminados_final.append({
            'id_producto_terminado': f'PT-{row["id_proceso"].split("-")[1]}', 'producto': row['producto'],
            'lote_produccion': row['lote'], 'linea_produccion': row['linea_produccion'],
            'cantidad_terminada': row['cantidad_terminada_wip'], 'fecha_termino_produccion': fecha_termino_real,
            'estado_almacen': estado_pt, 'costo_produccion_unitario': round(random.uniform(100, 1500), 2),
            'valor_venta_unitario': round(random.uniform(150, 2500), 2)
        })

df = pd.DataFrame(productos_terminados_final)
print("--- Datos generados correctamente. ---")


# --- 3. ANÁLISIS DE AUDITORÍA ---
print("--- Iniciando el análisis de auditoría... ---")
df['fecha_termino_produccion'] = pd.to_datetime(df['fecha_termino_produccion'], errors='coerce')

def reglas_auditoria(row):
    alertas = []
    if row['cantidad_terminada'] <= 0: alertas.append("Cantidad inválida")
    if row['costo_produccion_unitario'] <= 0: alertas.append("Costo inválido")
    if row['valor_venta_unitario'] <= 0: alertas.append("Valor de venta inválido")
    if row['costo_produccion_unitario'] > row['valor_venta_unitario']: alertas.append("Costo mayor al valor de venta")
    return " | ".join(alertas) if alertas else "OK"
df['alerta_heuristica'] = df.apply(reglas_auditoria, axis=1)

features = ['cantidad_terminada', 'costo_produccion_unitario', 'valor_venta_unitario']
X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
modelo = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
df['anomaly'] = modelo.fit_predict(X_scaled)
df['resultado_auditoria'] = df['anomaly'].map({1: 'Normal', -1: 'Anómalo'})

print("\n🔍 Productos terminados con alertas o anomalías detectadas:")
print(df[df['resultado_auditoria'] == 'Anómalo'][['id_producto_terminado', 'producto', 'alerta_heuristica', 'resultado_auditoria']])

# --- MODIFICACIÓN 3: Cambiar plt.show() por plt.savefig() y plt.close() ---
print("\n--- Generando los 4 gráficos de salida solicitados... ---")

# Gráfico 1
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x='linea_produccion', y='costo_produccion_unitario')
plt.title('Costo de Producción por Línea de Producción')
plt.savefig(os.path.join(output_dir, '1_costo_por_linea.png'), bbox_inches='tight')
plt.close()
print("Gráfico 1 guardado en la carpeta 'resultados'.")

# Gráfico 2
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='costo_produccion_unitario', y='valor_venta_unitario', hue='resultado_auditoria', style='estado_almacen')
plt.title('Valor de Venta vs Costo de Producción')
plt.savefig(os.path.join(output_dir, '2_venta_vs_costo.png'), bbox_inches='tight')
plt.close()
print("Gráfico 2 guardado en la carpeta 'resultados'.")

# Gráfico 3
plt.figure(figsize=(8, 4))
sns.histplot(df['cantidad_terminada'], bins=10, kde=True, color='skyblue')
plt.title('Distribución de Cantidades Terminadas')
plt.savefig(os.path.join(output_dir, '3_distribucion_cantidades.png'), bbox_inches='tight')
plt.close()
print("Gráfico 3 guardado en la carpeta 'resultados'.")

# Gráfico 4
df['margen_unitario'] = df['valor_venta_unitario'] - df['costo_produccion_unitario']
top_margen = df.sort_values(by='margen_unitario', ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(data=top_margen, x='producto', y='margen_unitario', palette='viridis')
plt.title('Top 10 Productos con Mayor Margen Unitario')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '4_top_margen.png'), bbox_inches='tight')
plt.close()
print("Gráfico 4 guardado en la carpeta 'resultados'.")