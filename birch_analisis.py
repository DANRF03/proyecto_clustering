
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import numpy as np
from pathlib import Path
import json

print("\nAnálisis con Birch (n_clusters=None)")

# --- Configuración ---
archivo = "dataset_final_mejorado.csv"  
variables = ['Situacion_economica', 'Fuente_ingresos', 'Sentimiento_financiero',
             'Dias_ejercicio', 'Horas_sueno', 'Edad']
out_dir = Path("salidas_birch")
out_dir.mkdir(exist_ok=True)

# --- Carga de datos ---
df = pd.read_csv(archivo, encoding='utf-8')
print(f"Dataset cargado: {df.shape}")

# --- Separar variables ---
numericas = [v for v in variables if v in df.select_dtypes(include='number').columns]
categoricas = [v for v in variables if v not in numericas]

# --- Preprocesamiento ---
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler())
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore"))
])
ct = ColumnTransformer([
    ("num", num_pipe, numericas),
    ("cat", cat_pipe, categoricas)
])
X = ct.fit_transform(df)

# --- Clustering Birch con n_clusters=None ---
modelo = Birch(n_clusters=None)
labels = modelo.fit_predict(X)

# --- Métricas ---
metrics = {
    "clusters_detectados": int(len(set(labels))),
    "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    "davies_bouldin": float(davies_bouldin_score(X, labels))
}

# --- Guardar resultados ---
df_out = df.copy()
df_out["cluster_birch"] = labels
df_out.to_csv(out_dir / "dataset_final_birch.csv", index=False, encoding="utf-8")
(out_dir / "metricas_birch.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

# --- Visualización PCA ---
pca = PCA(n_components=2)
Xp = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(Xp[:, 0], Xp[:, 1], c=labels, cmap='viridis', s=18, alpha=0.85)
plt.title(f"Birch Clustering (clusters detectados={metrics['clusters_detectados']})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(out_dir / "birch_pca.png")
plt.show()

print(f"\nBirch listo. Clusters detectados = {metrics['clusters_detectados']}")
print(f" -> {out_dir/'dataset_final_birch.csv'}")
print(f" -> {out_dir/'metricas_birch.json'}")
print(f" -> {out_dir/'birch_pca.png'}")
