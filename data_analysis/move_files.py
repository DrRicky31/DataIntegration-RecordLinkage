import os
import shutil

# Imposta le directory
main_folder = "./homework"
output_folder = "./extracted_files"

# Crea la cartella di destinazione se non esiste
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Scorri solo le sottocartelle nella cartella principale
for dir_name in os.listdir(main_folder):
    dir_path = os.path.join(main_folder, dir_name)
    if os.path.isdir(dir_path):
        # Cerca file CSV o JSON nella sottocartella
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and (file.endswith(".csv") or file.endswith(".json") or file.endswith(".xls") or file.endswith(".jsonl")):
                # Sposta il file nella cartella di destinazione
                shutil.move(file_path, output_folder)

print("Operazione completata: file CSV e JSON estratti e salvati nella cartella 'extracted_files'.")
