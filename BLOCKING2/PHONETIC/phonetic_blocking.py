import pandas as pd
from metaphone import doublemetaphone
from rapidfuzz.fuzz import ratio

# Caricamento del dataset da un file CSV
input_file = "data_analysis/merged_dataset_with_similarity.csv"  # Sostituisci con il percorso del tuo file
output_file = "BLOCKING2/PHONETIC/phonetic_merged.csv"

print("Caricamento del dataset...")
df = pd.read_csv(input_file)

# Verifica che il dataset abbia la colonna 'name'
if "name" not in df.columns:
    raise ValueError("Il dataset deve contenere una colonna chiamata 'name'.")

print(f"Dataset caricato con {len(df)} record.")

# Funzione per calcolare il nome fonetico
def calculate_phonetic(name):
    if pd.isnull(name):  # Gestione di valori mancanti
        return None
    return doublemetaphone(name)[0]  # Primo risultato di Double Metaphone

# Aggiunta della colonna del nome fonetico
print("Calcolo dei nomi fonetici...")
df["phonetic_name"] = df["name"].apply(calculate_phonetic)

# Mostra un'anteprima dei dati dopo l'aggiunta della colonna fonetica
print("\nAnteprima dei dati con colonna fonetica aggiunta:")
print(df.head())

# Funzione per verificare la similarità fonetica
def are_similar(phonetic1, phonetic2, threshold=90):
    if phonetic1 is None or phonetic2 is None:
        return False
    return ratio(phonetic1, phonetic2) >= threshold

# Fusione dei record basata sul nome fonetico con similarità
print("Raggruppamento e fusione dei record in base alla similarità fonetica...")

processed_phonetics = set()
merged_records = []

for idx, row in df.iterrows():
    phonetic = row["phonetic_name"]
    if phonetic in processed_phonetics:
        continue  # Salta i fonetici già processati

    # Trova tutti i record con fonetici simili
    similar_records = df[df["phonetic_name"].apply(lambda x: are_similar(phonetic, x))]

    # Fusione dei record simili
    print(f"\nElaborazione gruppo fonetico: {phonetic} ({len(similar_records)} record trovati)")
    merged = {
        "ids": list(similar_records["id"]) if "id" in similar_records.columns else None,
        "name": "; ".join(similar_records["name"]),
        "phonetic_name": phonetic
    }
    # Fusione di altre colonne
    for col in df.columns:
        if col not in ["id", "name", "phonetic_name"]:
            if pd.api.types.is_numeric_dtype(similar_records[col]):
                merged[col] = similar_records[col].mean()  # Media per numeri
            else:
                merged[col] = "; ".join(similar_records[col].astype(str).unique())  # Concatenazione unica per stringhe
    
    # Aggiungi il risultato fuso e segna i phonetic_name come processati
    merged_records.append(merged)
    processed_phonetics.update(similar_records["phonetic_name"])

# Converti il risultato in un DataFrame
merged_df = pd.DataFrame(merged_records)

# Mostra un'anteprima del dataset fuso
print("\nAnteprima del dataset fuso:")
print(merged_df.head())

# Salvataggio del dataset fuso in un nuovo file CSV
print(f"Salvataggio del dataset fuso in: {output_file}...")
merged_df.to_csv(output_file, index=False)
print("Operazione completata!")
