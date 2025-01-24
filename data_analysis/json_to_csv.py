import json
import csv
import os

def convert_json_to_csv_folder(input_folder, output_folder):
    """
    Converte tutti i file JSON o JSONL in una cartella in file CSV.

    :param input_folder: Percorso della cartella contenente i file JSON/JSONL.
    :param output_folder: Percorso della cartella in cui salvare i file CSV.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.json') or filename.endswith('.jsonl'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.csv")

            with open(input_path, 'r', encoding='utf-8') as infile:
                # Prova a caricare il file come JSON Lines
                try:
                    records = [json.loads(line) for line in infile]
                except json.JSONDecodeError:
                    infile.seek(0)  # Ripristina il puntatore
                    records = json.load(infile)  # Carica come JSON standard

            # Ottieni le intestazioni dal primo elemento
            headers = set()
            for record in records:
                headers.update(record.keys())
            headers = list(headers)

            # Scrivi i dati in formato CSV
            with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(records)

            print(f"File convertito con successo: {output_path}")

# Esempio di utilizzo
input_folder_path = 'extracted_files'  # Cambia con il percorso della tua cartella di input
output_folder_path = 'extracted_files' # Cambia con il percorso della tua cartella di output
convert_json_to_csv_folder(input_folder_path, output_folder_path)
