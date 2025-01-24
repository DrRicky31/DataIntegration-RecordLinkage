import zipfile
import os

# Directory contenente i file ZIP
homework_dir = "Hw5_IDD/homework"

# Funzione per elaborare e estrarre tutti i file ZIP in una directory
def extract_zip_files(directory):
    """
    Estrae tutti i file ZIP in una directory.
    """
    results = []
    if not os.path.exists(directory):
        print(f"La directory {directory} non esiste.")
        return results

    for file in os.listdir(directory):
        if file.endswith(".zip"):
            file_path = os.path.join(directory, file)
            extract_path = os.path.join(directory, f"extracted_{os.path.splitext(file)[0]}")
            os.makedirs(extract_path, exist_ok=True)
            
            # Estrazione del file ZIP
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                    results.append(extract_path)  # Aggiungi il percorso di estrazione
            except zipfile.BadZipFile:
                print(f"Errore: {file} non è un file ZIP valido.")
                continue

    return results

# Esegui il processo di estrazione sulla directory
extracted_paths = extract_zip_files(homework_dir)

# Visualizza i percorsi delle cartelle estratte
if extracted_paths:
    print("\nFile ZIP estratti:")
    for path in extracted_paths:
        print(path)
else:
    print("\nNessun file ZIP trovato o estratto.")
