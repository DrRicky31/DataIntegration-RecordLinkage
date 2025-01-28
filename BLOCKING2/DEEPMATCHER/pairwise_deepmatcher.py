import pandas as pd
import deepmatcher as dm
from sklearn.model_selection import train_test_split
from fuzzywuzzy import fuzz  # Usato per calcolare la similarità tra stringhe

# Funzione per calcolare la similarità tra due stringhe
def calculate_similarity(name1, name2):
    return fuzz.ratio(name1, name2) / 100.0  # Converte il punteggio in un valore tra 0 e 1

# Carica i dati della groundtruth e del file da verificare
def load_data(groundtruth_path, file_to_check_path):
    groundtruth = pd.read_csv(groundtruth_path)
    file_to_check = pd.read_csv(file_to_check_path)
    return groundtruth, file_to_check

# Valuta i risultati rispetto alla groundtruth
def evaluate_file(groundtruth, file_to_check, similarity_threshold=0.75):
    groundtruth_pairs = {
        (row['Name1'].strip().lower(), row['Name2'].strip().lower()): row['Similarity']
        for _, row in groundtruth.iterrows()
    }

    file_to_check_pairs = {
        (row['Name1'].strip().lower(), row['Name2'].strip().lower()): row['Similarity']
        for _, row in file_to_check.iterrows()
    }

    file_to_check_pairs.update({(b, a): sim for (a, b), sim in file_to_check_pairs.items()})

    valid_contributions = 0
    total_contributions = 0

    for pair, similarity in groundtruth_pairs.items():
        reverse_pair = (pair[1], pair[0])
        if similarity == 1:
            if (pair in file_to_check_pairs) or \
               (reverse_pair in file_to_check_pairs):
                valid_contributions += 1
            total_contributions += 1
        elif similarity == 0:
            if (pair not in file_to_check_pairs) and \
               (reverse_pair not in file_to_check_pairs):
                valid_contributions += 1
            total_contributions += 1

    contribution_percentage = (valid_contributions / total_contributions) * 100 if total_contributions > 0 else 0
    return contribution_percentage

# Prepara i dati per DeepMatcher
def prepare_data_for_deepmatcher(input_path, output_path, train_path, test_path, validation_path):
    data = pd.read_csv(input_path)

    if data['name'].isnull().any():
        data['name'] = data['name'].fillna('')

    data['name_clean'] = data['name'].apply(lambda x: [name.strip().lower() for name in x.split(";")])
    data = data.explode('name_clean')

    pairs = []
    for cluster_id, group in data.groupby('cluster'):
        names = group['name_clean'].dropna().unique()
        for i, name1 in enumerate(names):
            for name2 in names[i + 1:]:
                if calculate_similarity(name1, name2) >= 0.75:
                    pairs.append({'id': f'{cluster_id}_{name1}_{name2}', 'left_name': name1, 'right_name': name2, 'label': '1'})
                else:
                    pairs.append({'id': f'{cluster_id}_{name1}_{name2}', 'left_name': name1, 'right_name': name2, 'label': '0'})
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(output_path, index=False)
    print(f"File salvato per DeepMatcher in: {output_path}")

    # Suddivide i dati in train, validation e test
    train, temp = train_test_split(pairs_df, test_size=0.4, random_state=42)
    validation, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"File DeepMatcher salvati: train ({train_path}), validation ({validation_path}), test ({test_path})")


# Esegui DeepMatcher
def run_deepmatcher(train_path, validation_path, test_path, output_path):
    train, validation, test = dm.data.process(path='', train=train_path, validation=validation_path, test=test_path)

    model = dm.MatchingModel(attr_summarizer='hybrid')
    model.run_train(train, validation, best_save_path='best_model.pth', epochs=5) 

    predictions = model.run_prediction(test)

    # Prepara il DataFrame per l'output
    output_df = pd.DataFrame({
        'Name1': predictions['left_name'],  # Colonna disponibile per il nome a sinistra
        'Name2': predictions['right_name'],  # Colonna disponibile per il nome a destra
        'Similarity': predictions['match_score']  # Score di corrispondenza
    })

    # Salva il risultato in formato CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predizioni DeepMatcher salvate in: {output_path}")


def run_deepmatcher_with_model(test_path, model_path, output_path):
    # Carica i dati di test non etichettati, disabilitando la cache
    test = dm.data.process(path='', test=test_path, cache=False)
    
    # Carica il modello pre-addestrato
    model = dm.MatchingModel(attr_summarizer='hybrid')
    model.load_state(model_path)

    # Esegui le predizioni
    predictions = model.run_prediction(test, output_attributes=True)

    # Debug: Stampa le colonne di `predictions`
    print("Colonne disponibili in predictions:", predictions.columns)

    # Prepara il DataFrame per l'output
    output_df = pd.DataFrame({
        'Name1': predictions['left_name'],  # Colonna disponibile per il nome a sinistra
        'Name2': predictions['right_name'],  # Colonna disponibile per il nome a destra
        'Similarity': predictions['match_score']  # Score di corrispondenza
    })

    # Salva il risultato in formato CSV
    output_df.to_csv(output_path, index=False)
    print(f"Predizioni DeepMatcher salvate in: {output_path}")


# Funzione principale
def main():
    # Percorsi
    groundtruth_path = 'BLOCKING2/GROUNDTRUTH/groundtruth.csv'

    embedding_file_path = 'BLOCKING2/EMBEDDING/embedding_merged.csv'
    phonetic_file_path='BLOCKING2/PHONETIC/phonetic_merged.csv'

    embedding_input='BLOCKING2/DEEPMATCHER/embedding_input.csv'
    phonetic_input='BLOCKING2/DEEPMATCHER/phonetic_input.csv'

    train_path = 'BLOCKING2/DEEPMATCHER/train.csv'
    validation_path = 'BLOCKING2/DEEPMATCHER/validation.csv'
    test_path = 'BLOCKING2/DEEPMATCHER/test.csv'

    #deepmatcher_output_path = 'BLOCKING2/PHONETIC/deepmatcher_ph_results.csv'
    deepmatcher_output_path = 'BLOCKING2/EMBEDDING/deepmatcher_emb_results.csv'

    # Prepara i dati per DeepMatcher
    prepare_data_for_deepmatcher(embedding_file_path, embedding_input, train_path, test_path, validation_path)

    # Esegui DeepMatcher
    #run_deepmatcher_with_model(phonetic_input, model_path='best_model.pth', deepmatcher_output_path)

    run_deepmatcher(train_path, validation_path, test_path, deepmatcher_output_path)

    # Valuta i risultati
    groundtruth, deepmatcher_results = load_data(groundtruth_path, deepmatcher_output_path)
    contribution_percentage = evaluate_file(groundtruth, deepmatcher_results)

    print(f"Percentuale di contributi validi con DeepMatcher: {contribution_percentage:.2f}%")

if __name__ == "__main__":
    main()
