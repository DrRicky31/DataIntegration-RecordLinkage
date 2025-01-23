import pandas as pd
import glob
from rapidfuzz import fuzz
from rapidfuzz import process

# Funzione per mappare i nomi delle colonne in base allo schema mediato
def map_columns(schema_path, datasets_folder):
    print("Inizio mappatura delle colonne...")
    schema = pd.read_csv(schema_path)
    column_mapping = schema.set_index('Unified Attribute').to_dict()
    dataset_files = glob.glob(f"{datasets_folder}/*")

    datasets = []

    for dataset_path in dataset_files:
        dataset_name = dataset_path.split("\\")[-1]
        if dataset_name in column_mapping:
            print(f"Processando dataset: {dataset_name}...")
            dataset = pd.read_csv(dataset_path, encoding='latin1', on_bad_lines='skip')
            rename_dict = {v: k for k, v in column_mapping[dataset_name].items() if not pd.isna(v)}
            dataset = dataset.rename(columns=rename_dict)
            if "name" in dataset.columns:
                datasets.append(dataset)
            else:
                print(f"Il dataset '{dataset_name}' non contiene la colonna 'name' dopo la mappatura. Ignorato.")
        else:
            print(f"Mappatura non trovata per il dataset: {dataset_name}")
    print("Mappatura completata.")
    return datasets

# Funzione per unire i dataset usando "name" come chiave
def merge_datasets(datasets):
    print("Inizio unione dei dataset...")
    merged_df = pd.DataFrame()

    for idx, dataset in enumerate(datasets):
        print(f"Unendo dataset {idx + 1}/{len(datasets)}...")
        if merged_df.empty:
            merged_df = dataset
        else:
            merged_df = pd.merge(
                merged_df,
                dataset,
                on='name',
                how='outer',
                suffixes=('_left', '_right')
            )
            for col in merged_df.columns:
                if col.endswith('_left') or col.endswith('_right'):
                    base_col = col.replace('_left', '').replace('_right', '')
                    if base_col in merged_df.columns:
                        merged_df[base_col] = merged_df[base_col].combine_first(merged_df[col])
                        merged_df.drop(columns=col, inplace=True)
                    else:
                        merged_df.rename(columns={col: base_col}, inplace=True)
    print("Unione dei dataset completata.")
    return merged_df

# Funzione per unire i record simili basandosi su una soglia di similarità
def consolidate_similar_records(df, similarity_threshold=70):
    print("Inizio consolidamento dei record simili...")
    consolidated = []
    seen = set()
    total_records = len(df)
    print(f"Totale record da consolidare: {total_records}")

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Consolidando il record {i}/{total_records}...")

        if row["name"] in seen:
            continue
        similar_records = process.extract(
            row["name"], df["name"], scorer=fuzz.ratio, limit=None
        )
        similar_indices = [
            idx for name, score, idx in similar_records if score >= similarity_threshold
        ]
        seen.update(df.iloc[similar_indices]["name"])

        # Consolida i record trovati
        consolidated_record = df.iloc[similar_indices].fillna("").agg(
            lambda x: x.iloc[0]
            if x.nunique() == 1
            else "; ".join(map(str, x.unique()))
        )
        consolidated.append(consolidated_record)

    print("Consolidamento dei record simili completato.")
    return pd.DataFrame(consolidated)

# Percorsi principali
schema_path = 'MEDIATED_SCHEMA/unified_attribute_mapping.csv'
datasets_folder = 'extracted_files'

# Esegui la mappatura e unisci i dataset
mapped_datasets = map_columns(schema_path, datasets_folder)
merged_dataset = merge_datasets(mapped_datasets)

# Consolida i record con nomi simili
final_dataset = consolidate_similar_records(merged_dataset)

# Salva il dataset finale
final_dataset.to_csv('merged_dataset_with_similarity.csv', index=False)
print("Merge completato e consolidamento dei record simili. File salvato come 'merged_dataset_with_similarity.csv'.")
