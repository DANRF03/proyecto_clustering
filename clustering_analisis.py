import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print(" INICIANDO ANÁLISIS K-MEANS CON DATOS LIMPIOS")

CONFIG = {
    'random_state': 42,
    'n_init': 15,
    'max_k': 8,
    'min_k': 2,
    'figsize_large': (15, 10),
    'figsize_medium': (12, 8),
    'figsize_small': (10, 6),
    'palette': 'viridis'
}

# CARGA DIRECTA DE DATOS LIMPIOS
def cargar_datos_limpios():
    """Cargar el dataset ya limpio"""
    archivo_limpio = "dataset_final_mejorado.csv"
    
    print(f" Cargando {archivo_limpio}...")
    try:
        df = pd.read_csv(archivo_limpio, encoding='utf-8')
        print(f" Datos limpios cargados: {df.shape}")
        return df
    except Exception as e:
        print(f" Error cargando datos limpios: {e}")
        return None

# Cargar datos limpios
df_completo = cargar_datos_limpios()

if df_completo is None:
    print(" No se pudo cargar el archivo de datos limpios")
    exit()

# Mostrar información del dataset limpio
print(f"\n INFORMACIÓN DEL DATASET LIMPIO:")
print(f"   • Registros: {len(df_completo)}")
print(f"   • Columnas: {list(df_completo.columns)}")
print(f"   • Tipos de datos:")
print(df_completo.dtypes)
print(f"\n PRIMERAS FILAS:")
print(df_completo.head())

# ANÁLISIS EXPLORATORIO RÁPIDO


print("ANÁLISIS EXPLORATORIO DE DATOS LIMPIOS")


# Estadísticas descriptivas
print("\n ESTADÍSTICAS DESCRIPTIVAS:")
print(df_completo.describe())

# Distribución de variables categóricas
categorical_vars = ['Situacion_economica', 'Fuente_ingresos', 'Sentimiento_financiero', 'Actividad_recreativa']
print(f"\n DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS:")
for var in categorical_vars:
    if var in df_completo.columns:
        print(f"\n{var}:")
        print(df_completo[var].value_counts())

# PREPARACIÓN PARA K-MEANS


print("PREPARANDO DATOS PARA K-MEANS")


# Seleccionar variables para clustering
variables_kmeans = [
    'Situacion_economica', 'Fuente_ingresos', 'Sentimiento_financiero',
    'Dias_ejercicio', 'Horas_sueno', 'Edad'
]

# Verificar que todas las variables existan
variables_disponibles = [var for var in variables_kmeans if var in df_completo.columns]
print(f" Variables disponibles para clustering: {variables_disponibles}")

# Preparar datos para ML
df_ml = df_completo[variables_disponibles].copy()

# Identificar y codificar variables categóricas automáticamente
categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()

print(f"   • Variables categóricas: {categorical_cols}")
print(f"   • Variables numéricas: {numerical_cols}")

# Codificar variables categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le
    print(f"    {col} codificada ({len(le.classes_)} categorías)")

# Estandarizar datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_ml)
df_scaled = pd.DataFrame(df_scaled, columns=df_ml.columns)

print(" Datos preparados para K-Means")

# MÉTODO DEL CODO MEJORADO


print("1. ENCONTRANDO NÚMERO ÓPTIMO DE CLUSTERS")


inertias = []
silhouette_scores = []
calinski_scores = []
davies_scores = []
k_range = range(CONFIG['min_k'], CONFIG['max_k'] + 1)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=CONFIG['random_state'], n_init=CONFIG['n_init'])
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    
    inertias.append(kmeans.inertia_)
    
    if len(set(labels)) > 1:
        silhouette_scores.append(silhouette_score(df_scaled, labels))
        calinski_scores.append(calinski_harabasz_score(df_scaled, labels))
        davies_scores.append(davies_bouldin_score(df_scaled, labels))
    else:
        silhouette_scores.append(0)
        calinski_scores.append(0)
        davies_scores.append(float('inf'))

# Encontrar k óptimo usando múltiples métricas
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
optimal_k_calinski = k_range[np.argmax(calinski_scores)]
davies_scores_clean = [x if x != float('inf') else np.nan for x in davies_scores]
optimal_k_davies = k_range[np.nanargmin(davies_scores_clean)]

# Decisión final del k óptimo (priorizando silhouette)
optimal_k = optimal_k_silhouette
print(f" Número óptimo de clusters: {optimal_k}")
print(f"   • Silhouette: k={optimal_k_silhouette} (score: {max(silhouette_scores):.3f})")
print(f"   • Calinski-Harabasz: k={optimal_k_calinski} (score: {max(calinski_scores):.1f})")
print(f"   • Davies-Bouldin: k={optimal_k_davies} (score: {min(davies_scores_clean):.3f})")

# Graficar métricas de evaluación
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=CONFIG['figsize_large'])

# Inercia
ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
ax1.set_title('Método del Codo - Inercia')
ax1.set_xlabel('Número de Clusters (k)')
ax1.set_ylabel('Inercia')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'k óptimo = {optimal_k}')
ax1.legend()

# Silhouette
ax2.plot(k_range, silhouette_scores, marker='s', color='red', linewidth=2, markersize=8)
ax2.set_title('Puntuación de Silueta')
ax2.set_xlabel('Número de Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
ax2.axvline(x=optimal_k_silhouette, color='blue', linestyle=':', alpha=0.5)

# Calinski-Harabasz
ax3.plot(k_range, calinski_scores, marker='^', color='green', linewidth=2, markersize=8)
ax3.set_title('Índice Calinski-Harabasz')
ax3.set_xlabel('Número de Clusters (k)')
ax3.set_ylabel('Calinski-Harabasz Score')
ax3.grid(True, alpha=0.3)
ax3.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
ax3.axvline(x=optimal_k_calinski, color='blue', linestyle=':', alpha=0.5)

# Davies-Bouldin
ax4.plot(k_range, davies_scores_clean, marker='d', color='purple', linewidth=2, markersize=8)
ax4.set_title('Índice Davies-Bouldin')
ax4.set_xlabel('Número de Clusters (k)')
ax4.set_ylabel('Davies-Bouldin Score')
ax4.grid(True, alpha=0.3)
ax4.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7)
ax4.axvline(x=optimal_k_davies, color='blue', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('kmeans_metricas_optimal_k.png', dpi=150, bbox_inches='tight')
plt.close()

print(f" Gráfico de métricas guardado")

# APLICAR K-MEANS CON K ÓPTIMO


print(f"2. APLICANDO K-MEANS CON K = {optimal_k}")


kmeans_final = KMeans(n_clusters=optimal_k, random_state=CONFIG['random_state'], n_init=CONFIG['n_init'])
kmeans_labels = kmeans_final.fit_predict(df_scaled)

# Añadir clusters al dataframe completo
df_completo['Cluster_KMeans'] = kmeans_labels

# Métricas de evaluación finales
silhouette_avg = silhouette_score(df_scaled, kmeans_labels)
calinski_avg = calinski_harabasz_score(df_scaled, kmeans_labels)
davies_avg = davies_bouldin_score(df_scaled, kmeans_labels)

print(f" K-Means completado")
print(f"   • Clusters creados: {optimal_k}")
print(f"   • Silhouette Score: {silhouette_avg:.3f}")
print(f"   • Calinski-Harabasz: {calinski_avg:.1f}")
print(f"   • Davies-Bouldin: {davies_avg:.3f}")
print(f"   • Inercia final: {kmeans_final.inertia_:.2f}")


# PERFILAMIENTO DETALLADO DE CLUSTERS


print("3. PERFILAMIENTO DE CLUSTERS")

def crear_perfil_detallado(df, cluster_col):
    """Crear perfil detallado de clusters"""
    perfil = {}
    
    # Para variables categóricas
    categorical_vars = ['Situacion_economica', 'Fuente_ingresos', 'Sentimiento_financiero', 'Actividad_recreativa']
    for var in categorical_vars:
        if var in df.columns:
            agrupado = df.groupby(cluster_col)[var].agg([
                ('moda', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'N/A'),
                ('frecuencia', lambda x: x.value_counts().iloc[0] if len(x.value_counts()) > 0 else 0),
                ('total', 'count')
            ])
            agrupado['porcentaje'] = (agrupado['frecuencia'] / agrupado['total'] * 100).round(1)
            perfil[var] = agrupado[['moda', 'porcentaje']]
    
    # Para variables numéricas
    numerical_vars = ['Dias_ejercicio', 'Horas_sueno', 'Edad', 'Semestre']
    for var in numerical_vars:
        if var in df.columns:
            perfil[var] = df.groupby(cluster_col)[var].agg(['mean', 'std', 'min', 'max']).round(2)
    
    # Conteo de estudiantes por cluster
    perfil['conteo'] = df.groupby(cluster_col).size()
    
    return perfil

# Crear perfil detallado
perfil_detallado = crear_perfil_detallado(df_completo, 'Cluster_KMeans')

print("\n PERFIL DETALLADO DE CLUSTERS:")
for cluster_id in sorted(df_completo['Cluster_KMeans'].unique()):
    cluster_data = df_completo[df_completo['Cluster_KMeans'] == cluster_id]
    n_estudiantes = len(cluster_data)
    porcentaje = (n_estudiantes / len(df_completo)) * 100
    
    print(f"\n CLUSTER {cluster_id} ({n_estudiantes} estudiantes, {porcentaje:.1f}%):")
    
    # Variables clave
    print(f"   • Situación económica: {cluster_data['Situacion_economica'].mode()[0]}")
    print(f"   • Fuente ingresos: {cluster_data['Fuente_ingresos'].mode()[0]}")
    print(f"   • Sentimiento financiero: {cluster_data['Sentimiento_financiero'].mode()[0]}")
    print(f"   • Actividad recreativa: {cluster_data['Actividad_recreativa'].mode()[0]}")
    print(f"   • Ejercicio/semana: {cluster_data['Dias_ejercicio'].mean():.1f} días")
    print(f"   • Horas sueño: {cluster_data['Horas_sueno'].mean():.1f} horas")
    print(f"   • Edad promedio: {cluster_data['Edad'].mean():.1f} años")

# VISUALIZACIONES MEJORADAS


print("4. GENERANDO VISUALIZACIONES")

plt.style.use('default')
sns.set_palette(CONFIG['palette'])

# 1. Visualización PCA
pca = PCA(n_components=2, random_state=CONFIG['random_state'])
principal_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = kmeans_labels

plt.figure(figsize=CONFIG['figsize_medium'])
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], 
                     cmap=CONFIG['palette'], alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.title(f'K-Means Clustering - {optimal_k} Clusters de Estudiantes\n'
          f'({pca.explained_variance_ratio_.sum():.1%} varianza explicada)')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%})')

# Añadir centroides
centroids_pca = pca.transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, 
           c='red', edgecolors='black', linewidth=2, label='Centroides')

# Añadir anotaciones
for cluster in sorted(pca_df['Cluster'].unique()):
    cluster_data = pca_df[pca_df['Cluster'] == cluster]
    percentage = (len(cluster_data) / len(pca_df)) * 100
    plt.annotate(f'Cluster {cluster}\n({percentage:.1f}%)',
                (cluster_data['PC1'].mean(), cluster_data['PC2'].mean()),
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters_pca.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Visualización t-SNE (alternativa)
tsne = TSNE(n_components=2, random_state=CONFIG['random_state'], perplexity=min(30, len(df_scaled)-1))
tsne_components = tsne.fit_transform(df_scaled)
tsne_df = pd.DataFrame(tsne_components, columns=['TSNE1', 'TSNE2'])
tsne_df['Cluster'] = kmeans_labels

plt.figure(figsize=CONFIG['figsize_medium'])
scatter = plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df['Cluster'], 
                     cmap=CONFIG['palette'], alpha=0.7, s=60, edgecolors='w', linewidth=0.5)
plt.colorbar(scatter, label='Cluster')
plt.title(f'K-Means Clustering - {optimal_k} Clusters (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_clusters_tsne.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Distribución de clusters
plt.figure(figsize=CONFIG['figsize_small'])
cluster_counts = df_completo['Cluster_KMeans'].value_counts().sort_index()
colors = plt.cm.get_cmap(CONFIG['palette'])(np.linspace(0, 1, len(cluster_counts)))

bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors, alpha=0.7)
plt.title('Distribución de Estudiantes por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Número de Estudiantes')

for bar, count in zip(bars, cluster_counts.values):
    porcentaje = (count / len(df_completo)) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{count}\n({porcentaje:.1f}%)', ha='center', va='bottom', fontsize=9)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('kmeans_distribucion_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Heatmap de características por cluster
plt.figure(figsize=(12, 8))
cluster_means = df_completo.groupby('Cluster_KMeans')[numerical_cols].mean()
sns.heatmap(cluster_means.T, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5)
plt.title('Características Promedio por Cluster (Variables Numéricas)')
plt.tight_layout()
plt.savefig('kmeans_heatmap_caracteristicas.png', dpi=150, bbox_inches='tight')
plt.close()

print(" Visualizaciones guardadas")

# PROPUESTAS DE INTERVENCIÓN BASADAS EN DATOS


print("5. PROPUESTAS DE INTERVENCIÓN POR CLUSTER")

# Generar propuestas basadas en el perfil real de cada cluster
propuestas_personalizadas = {}

for cluster_id in sorted(df_completo['Cluster_KMeans'].unique()):
    cluster_data = df_completo[df_completo['Cluster_KMeans'] == cluster_id]
    
    # Analizar características del cluster
    situacion = cluster_data['Situacion_economica'].mode()[0]
    sentimiento = cluster_data['Sentimiento_financiero'].mode()[0]
    ejercicio = cluster_data['Dias_ejercicio'].mean()
    sueño = cluster_data['Horas_sueno'].mean()
    
    # Generar propuesta basada en el perfil
    propuesta = []
    
    if situacion in ['Mala', 'Complicada/Mala']:
        propuesta.append(" Asesoría económica y becas")
    if 'Ansiedad' in sentimiento:
        propuesta.append(" Talleres de manejo de estrés")
    if ejercicio < 2:
        propuesta.append(" Programa de actividad física")
    if sueño < 6:
        propuesta.append(" Educación sobre higiene del sueño")
    if 'Preocupación' in sentimiento:
        propuesta.append(" Orientación financiera personalizada")
    
    # Propuesta base si no hay necesidades específicas
    if not propuesta:
        propuesta.append(" Programa de desarrollo integral")
    
    propuestas_personalizadas[cluster_id] = propuesta

# Mostrar propuestas
for cluster_id, intervenciones in propuestas_personalizadas.items():
    cluster_data = df_completo[df_completo['Cluster_KMeans'] == cluster_id]
    n_estudiantes = len(cluster_data)
    
    print(f"\n CLUSTER {cluster_id} ({n_estudiantes} estudiantes):")
    print(f"   • Perfil: {cluster_data['Situacion_economica'].mode()[0]} situación, "
          f"{cluster_data['Sentimiento_financiero'].mode()[0]} sentimiento, "
          f"{cluster_data['Dias_ejercicio'].mean():.1f} días ejercicio, "
          f"{cluster_data['Horas_sueno'].mean():.1f} horas sueño")
    print(f"   • Intervenciones recomendadas:")
    for i, intervencion in enumerate(intervenciones, 1):
        print(f"      {i}. {intervencion}")

# GUARDAR RESULTADOS COMPLETOS

print("6. GUARDANDO RESULTADOS")

# Guardar datos con clusters
df_completo.to_csv('datos_con_clusters_final.csv', index=False, encoding='utf-8-sig')

# Guardar perfilamiento detallado
perfil_resumen = df_completo.groupby('Cluster_KMeans').agg({
    'Situacion_economica': lambda x: x.mode()[0],
    'Fuente_ingresos': lambda x: x.mode()[0],
    'Sentimiento_financiero': lambda x: x.mode()[0],
    'Dias_ejercicio': 'mean',
    'Horas_sueno': 'mean',
    'Edad': 'mean',
    'Numero_cuenta': 'count'
}).rename(columns={'Numero_cuenta': 'N_Estudiantes'}).round(2)

perfil_resumen.to_csv('perfil_clusters_detallado.csv', encoding='utf-8-sig')

print(" ARCHIVOS GUARDADOS:")
print("   - datos_con_clusters_final.csv")
print("   - perfil_clusters_detallado.csv")
print("   - kmeans_metricas_optimal_k.png")
print("   - kmeans_clusters_pca.png")
print("   - kmeans_clusters_tsne.png")
print("   - kmeans_distribucion_clusters.png")
print("   - kmeans_heatmap_caracteristicas.png")


print("ANÁLISIS K-MEANS COMPLETADO")

print(f"RESUMEN FINAL:")
print(f"   • Clusters óptimos: {optimal_k}")
print(f"   • Silhouette Score: {silhouette_avg:.3f}")
print(f"   • Estudiantes analizados: {len(df_completo)}")
print(f"   • Variables utilizadas: {len(variables_disponibles)}")
print(f"   • Clusters con intervenciones personalizadas: {len(propuestas_personalizadas)}")