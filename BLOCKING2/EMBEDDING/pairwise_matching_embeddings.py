import pandas as pd
from itertools import combinations
from jellyfish import jaro_winkler_similarity

# Caricamento del dataset
data = pd.read_csv('BLOCKING2/EMBEDDING/embedding_merged.csv', low_memory=False)

# Verifica se la colonna 'name' contiene NaN
if data['name'].isnull().any():
    print("Ci sono valori NaN nella colonna 'name'.")
else:
    print("La colonna 'name' non contiene valori NaN.")

# Funzione per separare i nomi separati da ";" e trasformarli in lowercase
def split_and_clean(name):
    names = name.split(";")
    return [name.strip().lower() for name in names]  # Rimuove gli spazi extra e converte in minuscolo

# Normalizzazione dei nomi per migliorare la qualità del matching
data['name_clean'] = data['name'].fillna('').apply(split_and_clean)

# Creazione di una lista di tutte le coppie per ogni cluster
all_matches = []

# Loop sui cluster
for cluster_id, group in data.groupby('cluster'):
    # Estrai tutti i nomi puliti per il cluster corrente
    names = group['name_clean'].explode().dropna().unique()
    
    # Genera tutte le combinazioni di coppie di nomi
    for name1, name2 in combinations(names, 2):
        # Calcola la similarità tra i due nomi
        similarity = jaro_winkler_similarity(name1, name2)
        all_matches.append({'Cluster': cluster_id, 'Name1': name1, 'Name2': name2, 'Similarity': similarity})

# Converti i risultati in un DataFrame
matches_df = pd.DataFrame(all_matches)

# Filtro dei match con una soglia di similarità
threshold = 0.50  # Soglia di similarità
matches_df = matches_df[matches_df['Similarity'] > threshold]

# Verifica se ci sono match
if matches_df.empty:
    print("Nessun match trovato.")
else:
    print(f"Trovati {len(matches_df)} match con similarità maggiore di {threshold}.")
    
    # Salvataggio su CSV
    matches_df.to_csv('BLOCKING2/EMBEDDING/embedding_matching_results.csv', index=False)

    print("Matching completato. Risultati salvati in 'embedding_matching_results.csv'.")
