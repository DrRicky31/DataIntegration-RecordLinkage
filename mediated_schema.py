import os
import pandas as pd
import json

# Funzione per leggere i file CSV
def read_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')

# Funzione per leggere i file JSON
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.json_normalize(data)

# Funzione per leggere i file JSONL
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return pd.json_normalize(data)

# Funzione per leggere i file Excel (supporta sia .xls che .xlsx)
def read_excel(file_path):
    if file_path.endswith('.xls'):
        return pd.read_excel(file_path, engine='xlrd')
    else:
        return pd.read_excel(file_path, engine='openpyxl')

# Funzione per creare uno schema mediato e salvarlo in un file Excel
def create_mediated_schema(directory_path, output_file="mediated_schema.xlsx"):
    schema = {}

    # Controllo se la directory esiste
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} non trovata.")
        return schema

    # Scorre tutti i file nella directory e nelle sottocartelle
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.startswith("._"):
                continue  # Ignora i file di sistema

            file_path = os.path.join(root, file)

            try:
                if file.endswith('.csv'):
                    df = read_csv(file_path)
                elif file.endswith('.json'):
                    df = read_json(file_path)
                elif file.endswith('.jsonl'):
                    df = read_jsonl(file_path)
                elif file.endswith('.xls') or file.endswith('.xlsx'):
                    df = read_excel(file_path)
                else:
                    continue

                for column in df.columns:
                    if column not in schema:
                        schema[column] = []
                    schema[column].append(file_path)

            except Exception as e:
                print(f"Errore durante la lettura del file {file_path}: {e}")

    # Converti lo schema in un DataFrame
    schema_df = pd.DataFrame([(key, ', '.join(value)) for key, value in schema.items()], columns=['Attributo', 'Provenienza'])

    # Salva lo schema mediato in un file Excel
    schema_df.to_excel(output_file, index=False)

    print(f"Schema mediato salvato in {output_file}")
    return schema_df

# Funzione principale
if __name__ == "__main__":
    directory_path = "./extracted_files"
    output_file = "mediated_schema.xlsx"
    create_mediated_schema(directory_path, output_file)
