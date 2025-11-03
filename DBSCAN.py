import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

print("\n🔍 Análisis DBSCAN con datos limpios")
print("=" * 60)

CONFIG = {
    'eps': 0.8,  # Distancia máxima entre puntos del mismo cluster
    'min_samples': 5,  # Puntos mínimos para formar un cluster
    'random_state': 42,
    'palette': 'viridis',
    'figsize': (12, 8)
}

# Carga de datos
archivo = "dataset_final_mejorado.csv"
df = pd.read_csv(archivo, encoding='utf-8')
print(f"✅ Dataset cargado: {df.shape}")

# Variables a usar
variables = ['Situacion_economica', 'Fuente_ingresos', 'Sentimiento_financiero',
             'Dias_ejercicio', 'Horas_sueno', 'Edad']

X = df[variables].select_dtypes(include='number')

# DBSCAN
modelo = DBSCAN(eps=CONFIG['eps'], min_samples=CONFIG['min_samples'])
df['Cluster_DBSCAN'] = modelo.fit_predict(X)

n_clusters = len(set(modelo.labels_)) - (1 if -1 in modelo.labels_ else 0)
print(f"📊 Clusters detectados: {n_clusters}")

# Visualización PCA
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(X), columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster_DBSCAN']

plt.figure(figsize=CONFIG['figsize'])
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', alpha=0.8)
plt.title(f"DBSCAN Clustering (eps={CONFIG['eps']}, min_samples={CONFIG['min_samples']})")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.savefig("dbscan_clusters_pca.png")
plt.show()

# Exporta resultados
df.to_csv("datos_con_clusters_dbscan.csv", index=False, encoding='utf-8-sig')
print("\n✅ Resultados guardados en 'datos_con_clusters_dbscan.csv'")
print("✅ Gráfica: 'dbscan_clusters_pca.png'")
print("ANÁLISIS DBSCAN COMPLETADO ✅")