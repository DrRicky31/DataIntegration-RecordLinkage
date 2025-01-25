import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

# Caricamento del dataset da un file CSV
input_file = "data_analysis/merged_dataset_with_similarity.csv"  # Sostituisci con il percorso del tuo file
output_file = "BLOCKING2/EMBEDDING/embedding_merged.csv"

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

# Clustering con DBSCAN basato su embedding
print("Esecuzione del clustering con DBSCAN...")
dbscan = DBSCAN(eps=0.2, min_samples=1, metric="cosine")  # Ridotto eps a 0.1
clusters = dbscan.fit_predict(embeddings)

# Aggiungi i cluster come chiavi di blocco al DataFrame
df["cluster"] = clusters
print(f"\nClustering completato. Numero di cluster trovati: {len(set(clusters)) - (1 if -1 in clusters else 0)}")

# Fusione dei record all'interno di ciascun cluster
def merge_records(group):
    merged = {
        "ids": list(group["id"]) if "id" in group.columns else None,
        "name": "; ".join(group["name"]),  # Cambiato da ", " a "; "
        "cluster": group["cluster"].iloc[0],
    }
    # Fusione di altre colonne
    for col in group.columns:
        if col not in ["id", "name", "cluster"]:
            if pd.api.types.is_numeric_dtype(group[col]):
                merged[col] = group[col].mean()  # Media per numeri
            else:
                merged[col] = "; ".join(group[col].astype(str).unique())  # Cambiato da ", " a "; "
    return pd.Series(merged)

print("Raggruppamento e fusione dei record all'interno dei cluster...")

# Fusione dei cluster senza rimuovere quelli con un singolo elemento
merged_df = df.groupby("cluster").apply(merge_records).reset_index(drop=True)

# Mostra un'anteprima del dataset fuso
print("\nAnteprima del dataset fuso:")
print(merged_df.head())

# Salvataggio del dataset fuso in un nuovo file CSV
print(f"Salvataggio del dataset fuso in: {output_file}...")
merged_df.to_csv(output_file, index=False)
print("Operazione completata!")
