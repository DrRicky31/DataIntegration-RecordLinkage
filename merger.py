import pandas as pd
import glob

# Funzione per mappare i nomi delle colonne in base allo schema mediato
def map_columns(schema_path, datasets_folder):
    # Leggere lo schema mediato
    schema = pd.read_csv(schema_path)
    column_mapping = schema.set_index('Unified Attribute').to_dict()

    # Trovare tutti i file nella cartella dei dataset
    dataset_files = glob.glob(f"{datasets_folder}/*")

    datasets = []

    for dataset_path in dataset_files:
        dataset_name = dataset_path.split("\\")[-1]

        # Verificare se il dataset ha una mappatura nello schema
        if dataset_name in column_mapping:
            dataset = pd.read_csv(dataset_path, encoding='latin1', on_bad_lines='skip')  # Ignora righe problematiche

            # Rinominare le colonne in base al mapping
            rename_dict = {v: k for k, v in column_mapping[dataset_name].items() if not pd.isna(v)}
            dataset = dataset.rename(columns=rename_dict)

            # Verifica che il dataset contenga la colonna "name"
            if "name" in dataset.columns:
                datasets.append(dataset)
            else:
                print(f"Il dataset '{dataset_name}' non contiene la colonna 'name' dopo la mappatura. Ignorato.")
        else:
            print(f"Mappatura non trovata per il dataset: {dataset_name}")

    return datasets

# Funzione per unire i dataset usando "name" come chiave
def merge_datasets(datasets):
    merged_df = pd.DataFrame()

    for dataset in datasets:
        if merged_df.empty:
            merged_df = dataset
        else:
            # Effettua il merge con priorità al primo valore trovato
            merged_df = pd.merge(
                merged_df,
                dataset,
                on='name',
                how='outer',
                suffixes=('_left', '_right')
            )

            # Risolvi i conflitti mantenendo il primo valore
            for col in merged_df.columns:
                if col.endswith('_left') or col.endswith('_right'):
                    base_col = col.replace('_left', '').replace('_right', '')
                    if base_col in merged_df.columns:
                        merged_df[base_col] = merged_df[base_col].combine_first(merged_df[col])
                        merged_df.drop(columns=col, inplace=True)
                    else:
                        merged_df.rename(columns={col: base_col}, inplace=True)

    return merged_df

# Percorsi principali
schema_path = 'MEDIATED_SCHEMA/unified_attribute_mapping.csv'
datasets_folder = 'extracted_files'  # Modificare con il percorso reale

# Esegui la mappatura e unisci i dataset
mapped_datasets = map_columns(schema_path, datasets_folder)
final_dataset = merge_datasets(mapped_datasets)

# Salva il dataset finale
final_dataset.to_csv('merged_dataset.csv', index=False)
print("Merge completato. File salvato come 'merged_dataset.csv'.")
