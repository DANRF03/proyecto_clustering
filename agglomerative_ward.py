# -*- coding: utf-8 -*-
"""
Agglomerative (Ward) con visuales extendidas:
- Busca K óptimo por silhouette (2..10 por defecto)
- Guarda: etiquetas, perfiles, métricas y 5 figuras:
  * agglomerative_pca.png (scatter PCA 2D)
  * agglomerative_silhouette_k.png (silhouette vs K)
  * agglomerative_cluster_sizes.png (conteos por cluster)
  * agglomerative_profile_heatmap.png (heatmap de medias numéricas por cluster)
  * agglomerative_dendrogram.png (dendrograma Ward truncado)
"""

import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Para dendrograma
from scipy.cluster.hierarchy import linkage, dendrogram

warnings.filterwarnings("ignore", category=FutureWarning)

SCALERS = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler()
}

def _one_hot_encoder():
    """OneHotEncoder compatible con distintas versiones de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # >=1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # <1.2

def pca_scatter(X, labels, out_path: Path):
    if X.shape[1] < 2 or labels is None:
        return
    p = PCA(n_components=2, random_state=42)
    Xp = p.fit_transform(X)
    plt.figure()
    plt.scatter(Xp[:, 0], Xp[:, 1], c=labels, s=18, alpha=0.85)
    plt.title("Agglomerative (Ward) - PCA 2D")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_silhouette_curve(k_values, sil_scores, out_path: Path):
    plt.figure()
    plt.plot(k_values, sil_scores, marker='o')
    plt.title("Silhouette vs K (Agglomerative - Ward)")
    plt.xlabel("K"); plt.ylabel("Silhouette")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_cluster_sizes(labels, out_path: Path):
    uniq, counts = np.unique(labels, return_counts=True)
    plt.figure()
    plt.bar([str(int(u)) for u in uniq], counts)
    plt.title("Tamaño de clúster (Agglomerative - Ward)")
    plt.xlabel("Cluster"); plt.ylabel("Cantidad")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_profile_heatmap(df_out, labels, numericas, out_path: Path):
    if not numericas:
        return
    data = []
    clusters = sorted([c for c in np.unique(labels) if c != -1])
    for c in clusters:
        part = df_out[df_out["cluster_ward"] == c]
        row = [np.nanmean(part[col]) for col in numericas]
        data.append(row)
    mat = np.array(data)
    plt.figure()
    im = plt.imshow(mat, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.yticks(range(len(clusters)), [str(int(c)) for c in clusters])
    plt.xticks(range(len(numericas)), numericas, rotation=45, ha='right')
    plt.title("Heatmap de perfiles (medias de variables numéricas)")
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def perfiles(df, labels, numericas, categoricas):
    df2 = df.copy()
    df2["cluster_ward"] = labels
    filas = []
    for c in sorted([x for x in set(labels) if x != -1]):
        part = df2[df2["cluster_ward"] == c]
        row = {"cluster": int(c), "n": int(len(part))}
        for col in numericas:
            if col in part.columns:
                row[col + "_mean"] = float(np.nanmean(part[col]))
        for col in categoricas:
            if col in part.columns:
                vc = part[col].value_counts(normalize=True).head(3)
                row[col + "_top"] = ", ".join([f"{k} ({v*100:.1f}%)" for k, v in vc.items()])
        filas.append(row)
    return pd.DataFrame(filas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="dataset_final_mejorado.csv")
    ap.add_argument("--out", default="salidas")
    ap.add_argument("--scaler", default="standard", choices=["standard","minmax","robust"])
    ap.add_argument("--k-range", default="2:10")
    ap.add_argument("--num", default="Semestre,Dias_ejercicio,Horas_sueno,Edad")
    ap.add_argument("--cat", default="Situacion_economica,Fuente_ingresos,Sentimiento_financiero,Actividad_recreativa")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8")
    kmin, kmax = [int(x) for x in args.k_range.split(":")]

    numericas = [c.strip() for c in args.num.split(",") if c.strip() in df.columns]
    categoricas = [c.strip() for c in args.cat.split(",") if c.strip() in df.columns]

    for idc in ["Numero_cuenta", "Número de cuenta", "numero_cuenta"]:
        if idc in df.columns:
            df = df.drop(columns=[idc])

    for c in numericas:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", SCALERS[args.scaler])])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", _one_hot_encoder())])

    ct = ColumnTransformer([("num", num_pipe, numericas),
                            ("cat", cat_pipe, categoricas)],
                           remainder="drop")

    X = ct.fit_transform(df)

    # --- búsqueda de K
    k_values, sil_scores = [], []
    best_s, best_k, best_labels = -np.inf, None, None
    for k in range(kmin, kmax + 1):
        try:
            mdl = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labs = mdl.fit_predict(X)
            if len(set(labs)) > 1:
                s = silhouette_score(X, labs)
                k_values.append(k); sil_scores.append(s)
                if s > best_s:
                    best_s, best_k, best_labels = s, k, labs
        except Exception:
            pass

    # métricas adicionales
    if best_labels is not None and len(set(best_labels)) > 1:
        metrics = {
            "silhouette": float(best_s),
            "calinski_harabasz": float(calinski_harabasz_score(X, best_labels)),
            "davies_bouldin": float(davies_bouldin_score(X, best_labels)),
            "k_optimo": int(best_k)
        }
    else:
        metrics = {"silhouette": None, "k_optimo": None}

    # guardar etiquetas y perfiles
    df_out = pd.read_csv(args.input, encoding="utf-8").copy()
    df_out["cluster_ward"] = best_labels if best_labels is not None else -1
    df_out.to_csv(out_dir / "dataset_final_agglomerative.csv", index=False, encoding="utf-8")

    perf = perfiles(df_out, df_out["cluster_ward"].values, numericas, categoricas)
    perf.to_csv(out_dir / "perfiles_agglomerative.csv", index=False, encoding="utf-8")

    # --- figuras
    pca_scatter(X, best_labels, out_dir / "agglomerative_pca.png")
    if k_values:
        plot_silhouette_curve(k_values, sil_scores, out_dir / "agglomerative_silhouette_k.png")
    if best_labels is not None:
        plot_cluster_sizes(best_labels, out_dir / "agglomerative_cluster_sizes.png")
        plot_profile_heatmap(df_out, best_labels, numericas, out_dir / "agglomerative_profile_heatmap.png")

    # dendrograma (Ward) sobre X ya escalado/onehot
    try:
        Z = linkage(X, method="ward")
        plt.figure(figsize=(8, 4.5))
        dendrogram(Z, truncate_mode="lastp", p=min(10, len(X)//2), leaf_rotation=0)
        plt.title("Dendrograma (Ward) - truncado"); plt.xlabel("Grupos fusionados"); plt.ylabel("Distancia")
        plt.tight_layout(); plt.savefig(out_dir / "agglomerative_dendrogram.png"); plt.close()
    except Exception:
        pass  # si falla por tamaño/numérico, lo omitimos

    # métricas
    (out_dir / "metricas_agglomerative.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nAgglomerative (Ward) listo. K óptimo = {metrics.get('k_optimo')}, silhouette = {metrics.get('silhouette')}")
    print(f"   -> {out_dir/'dataset_final_agglomerative.csv'}")
    print(f"   -> {out_dir/'perfiles_agglomerative.csv'}")
    print(f"   -> {out_dir/'agglomerative_pca.png'}")
    print(f"   -> {out_dir/'agglomerative_silhouette_k.png'}")
    print(f"   -> {out_dir/'agglomerative_cluster_sizes.png'}")
    print(f"   -> {out_dir/'agglomerative_profile_heatmap.png'}")
    print(f"   -> {out_dir/'agglomerative_dendrogram.png'}")
    print(f"   -> {out_dir/'metricas_agglomerative.json'}")

if __name__ == "__main__":
    main()
