import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Carica il dataset
file_path = "BLOCKING2/EMBEDDING/embedding_merged.csv"  # Modifica con il percorso corretto
df = pd.read_csv(file_path)

# Divide i nomi e filtra i cluster con almeno 2 elementi
df["name"] = df["name"].apply(lambda x: x.split(";"))
df = df[df["name"].apply(len) > 1]

# Conta il numero di elementi per cluster
df["size"] = df["name"].apply(len)

# Ordina i cluster per grandezza e seleziona quelli dalla posizione 51 alla 80
top_clusters = df.groupby("cluster").sum().nlargest(80, "size").reset_index()
df_selected = df[df["cluster"].isin(top_clusters["cluster"].iloc[50:80])]

# Creiamo una lista di parole per ogni cluster
words = list(set([name for names in df_selected["name"] for name in names]))

# Carica il modello pre-addestrato
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Otteniamo i vettori embedding per ogni parola
word_vectors = {word: model.encode(word) for word in words}

# Riduciamo la dimensionalità con PCA prima di t-SNE
pca = PCA(n_components=20)  # Da 384D a 20D
vectors = np.array([word_vectors[w] for w in words])
vectors_pca = pca.fit_transform(vectors)

# Applichiamo t-SNE per ridurre a 2D
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
vectors_2d = tsne.fit_transform(vectors_pca)

# Creiamo un DataFrame per la visualizzazione
df_plot = pd.DataFrame(vectors_2d, columns=["x", "y"])
df_plot["word"] = words

# Assegniamo un cluster a ogni parola
def find_cluster(word):
    for cluster_id in df_selected["cluster"].unique():
        for names in df_selected[df_selected["cluster"] == cluster_id]["name"]:
            if word in names:
                return cluster_id
    return None

df_plot["cluster"] = df_plot["word"].apply(find_cluster)
df_plot.dropna(inplace=True)

# Aggiungi un solo nome per cluster (primo nome di ogni cluster)
cluster_names = {cluster_id: next(iter(df_selected[df_selected["cluster"] == cluster_id]["name"]))[0]
                 for cluster_id in df_selected["cluster"].unique()}

# Visualizzazione con Seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_plot, x="x", y="y", hue="cluster", palette="tab10", legend=False, alpha=0.7)

# Aggiungi un solo nome per cluster (etichetta)
for cluster_id, name in cluster_names.items():
    cluster_points = df_plot[df_plot["cluster"] == cluster_id]
    plt.text(cluster_points["x"].mean(), cluster_points["y"].mean(), name, fontsize=10, ha="center", va="center")

# Non aggiungere etichette per i nodi
plt.title("Clusterizzazione dei Word Embeddings con t-SNE", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
