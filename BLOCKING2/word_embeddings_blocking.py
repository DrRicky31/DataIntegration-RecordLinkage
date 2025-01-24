import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from umap import UMAP
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter

# Caricamento del dataset da un file CSV
input_file = "data_analysis/merged_dataset_with_similarity.csv"  # Sostituisci con il percorso del tuo file
output_file = "BLOCKING2/embedding_merged.csv"

print("Caricamento del dataset...")
df = pd.read_csv(input_file)

# Verifica che il dataset abbia la colonna 'name'
if "name" not in df.columns:
    raise ValueError("Il dataset deve contenere una colonna chiamata 'name'.")

print(f"Dataset caricato con {len(df)} record.")

# Calcolo degli embedding con Sentence Transformers
print("Calcolo degli embedding per i nomi...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Modello di embedding
embeddings = model.encode(df["name"].astype(str).tolist(), show_progress_bar=True)
print("Calcolo completato.")

# Riduzione della dimensionalità con UMAP
print("Riduzione della dimensionalità con UMAP...")
reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
reduced_embeddings = reducer.fit_transform(embeddings)
print("Riduzione completata.")

# Analisi della distribuzione delle distanze
print("Calcolo della distribuzione delle distanze...")
distances = pairwise_distances(reduced_embeddings, metric="euclidean")
flattened_distances = np.sort(distances, axis=None)
optimal_eps = np.percentile(flattened_distances, 5)  # 5° percentile come riferimento iniziale
print(f"Valore suggerito per eps: {optimal_eps:.4f}")

# Clustering con DBSCAN basato su embedding ridotti
print("Esecuzione del clustering con DBSCAN...")
dbscan = DBSCAN(eps=optimal_eps, min_samples=3, metric="euclidean")
clusters = dbscan.fit_predict(reduced_embeddings)

# Aggiungi i cluster come chiavi di blocco al DataFrame
df["cluster"] = clusters
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"\nClustering completato. Numero di cluster trovati: {num_clusters}")

# Fusione dei record all'interno di ciascun cluster
def merge_records(group):
    merged = {
        "ids": list(group["id"]) if "id" in group.columns else None,
        "name": ", ".join(group["name"]),
        "cluster": group["cluster"].iloc[0],
    }
    # Fusione di altre colonne
    for col in group.columns:
        if col not in ["id", "name", "cluster"]:
            if pd.api.types.is_numeric_dtype(group[col]):
                merged[col] = group[col].mean()  # Media per numeri
            else:
                merged[col] = ", ".join(group[col].astype(str).unique())  # Concatenazione unica per stringhe
    return pd.Series(merged)

print("Raggruppamento e fusione dei record all'interno dei cluster...")
merged_df = df.groupby("cluster").apply(merge_records).reset_index(drop=True)

# Mostra un'anteprima del dataset fuso
print("\nAnteprima del dataset fuso:")
print(merged_df.head())

# Salvataggio del dataset fuso in un nuovo file CSV
print(f"Salvataggio del dataset fuso in: {output_file}...")
merged_df.to_csv(output_file, index=False)
print("Operazione completata!")
