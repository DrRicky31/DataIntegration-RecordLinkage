import pandas as pd

# Carica i dati della groundtruth e del file da verificare
def load_data(groundtruth_path, file_to_check_path):
    groundtruth = pd.read_csv(groundtruth_path)
    file_to_check = pd.read_csv(file_to_check_path)
    return groundtruth, file_to_check

def evaluate_file(groundtruth, file_to_check, similarity_threshold=0.75):
    # Prepara i dati dalla groundtruth
    groundtruth_pairs = {
        (row['Name1'].strip().lower(), row['Name2'].strip().lower()): row['Similarity'] for _, row in groundtruth.iterrows()
    }

    # Prepara i dati dal file da controllare
    file_to_check_pairs = {
        (row['Name1'].strip().lower(), row['Name2'].strip().lower()): row['Similarity'] for _, row in file_to_check.iterrows()
    }

    # Aggiungi le permutazioni inverso (b, a)
    file_to_check_pairs.update({(b, a): sim for (a, b), sim in file_to_check_pairs.items()})

    # Inizializza i contatori
    total_contributions = 0
    valid_contributions = 0

    for pair, similarity in groundtruth_pairs.items():
        reverse_pair = (pair[1], pair[0])

        if similarity == 1:
            # Verifica se la coppia è presente nel file da controllare con Similarity > threshold
            if (pair in file_to_check_pairs and file_to_check_pairs[pair] > similarity_threshold) or \
               (reverse_pair in file_to_check_pairs and file_to_check_pairs[reverse_pair] > similarity_threshold):
                valid_contributions += 1
            total_contributions += 1
        elif similarity == 0:
            # Verifica se la coppia non è presente o ha Similarity < threshold
            if (pair not in file_to_check_pairs or file_to_check_pairs.get(pair, 0) < similarity_threshold) and \
               (reverse_pair not in file_to_check_pairs or file_to_check_pairs.get(reverse_pair, 0) < similarity_threshold):
                valid_contributions += 1
            total_contributions += 1

    # Calcola la percentuale di contributi validi
    contribution_percentage = (valid_contributions / total_contributions) * 100 if total_contributions > 0 else 0
    return contribution_percentage

# Funzione principale
def main():
    # Percorsi dei file
    groundtruth_path = 'BLOCKING2\GROUNDTRUTH\groundtruth.csv'  # Sostituisci con il percorso del file groundtruth
    embedding_file_path = 'BLOCKING2\EMBEDDING\embedding_matching_results.csv'  # Sostituisci con il percorso del file embedding
    phonetic_file_path = 'BLOCKING2\PHONETIC\phonetic_matching_results.csv'  # Sostituisci con il percorso del file phonetic

    # Carica i dati
    groundtruth, embedding_file = load_data(groundtruth_path, embedding_file_path)
    _, phonetic_file = load_data(groundtruth_path, phonetic_file_path)

    # Valuta il file embedding
    embedding_contribution_percentage = evaluate_file(groundtruth, embedding_file)
    print(f"Percentuale di contributi validi (Embedding): {embedding_contribution_percentage:.2f}%")

    # Valuta il file phonetic
    phonetic_contribution_percentage = evaluate_file(groundtruth, phonetic_file)
    print(f"Percentuale di contributi validi (Phonetic): {phonetic_contribution_percentage:.2f}%")

if __name__ == "__main__":
    main()
