import pandas as pd
import recordlinkage

# Carica i dati di blocking
attribute_based_file = '/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/BLOCKING/attribute_based_blocks.csv'
similarity_based_file = '/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/BLOCKING/similarity_based_blocks.csv'
groundtruth_file = '/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/GROUNDTRUTH/top_100_similar_datasets_by_attributes.csv'

# Carica i dati dai file CSV
attribute_based_blocks = pd.read_csv(attribute_based_file)
similarity_based_blocks = pd.read_csv(similarity_based_file)
groundtruth_df = pd.read_csv(groundtruth_file)

# Funzione per calcolare il pairwise matching
def calculate_pairwise_matching(blocks):
    # Creazione di un indicizzatore per il blocking
    indexer = recordlinkage.Index()
    indexer.block(left_on='Dataset 1', right_on='Dataset 2')

    # Generazione delle coppie candidate
    candidate_links = indexer.index(blocks)

    # Comparatore per confrontare gli attributi
    compare = recordlinkage.Compare()
    compare.exact('Dataset 1', 'Dataset 2', label='dataset_match')
    compare.numeric('Similar Attributes', 'Similar Attributes', label='attribute_match')
    compare.numeric('Similarity Ratio', 'Similarity Ratio', label='similarity_match')

    # Calcolo delle caratteristiche di similarità
    features = compare.compute(candidate_links, blocks)

    # Filtraggio delle coppie che superano una determinata soglia
    matches = features[features.sum(axis=1) > 1]

    return matches

# Calcola il matching per le due strategie di blocking
matches_attribute = calculate_pairwise_matching(attribute_based_blocks)
matches_similarity = calculate_pairwise_matching(similarity_based_blocks)

# Assicurarsi che i tipi di dati siano coerenti tra groundtruth e blocchi di dati
# Convertire le colonne in stringhe per evitare errori di tipo
groundtruth_df['Dataset 1'] = groundtruth_df['Dataset 1'].astype(str)
groundtruth_df['Dataset 2'] = groundtruth_df['Dataset 2'].astype(str)
similarity_based_blocks['Dataset 1'] = similarity_based_blocks['Dataset 1'].astype(str)
similarity_based_blocks['Dataset 2'] = similarity_based_blocks['Dataset 2'].astype(str)
attribute_based_blocks['Dataset 1'] = attribute_based_blocks['Dataset 1'].astype(str)
attribute_based_blocks['Dataset 2'] = attribute_based_blocks['Dataset 2'].astype(str)

# Trova le corrispondenze tra groundtruth e blocchi di dati (similarity/attribute)
similarity_matches = similarity_based_blocks.merge(
    groundtruth_df, on=['Dataset 1', 'Dataset 2'], how='inner'
)
attribute_matches = attribute_based_blocks.merge(
    groundtruth_df, on=['Dataset 1', 'Dataset 2'], how='inner'
)

# Conta il numero di corrispondenze trovate
similarity_match_count = len(similarity_matches)
attribute_match_count = len(attribute_matches)

# Calcola le metriche per il blocking basato sulla similarità
true_positive_similarity = similarity_match_count
false_positive_similarity = len(similarity_based_blocks) - true_positive_similarity
false_negative_similarity = len(groundtruth_df) - true_positive_similarity

precision_similarity = (
    true_positive_similarity / (true_positive_similarity + false_positive_similarity)
    if (true_positive_similarity + false_positive_similarity) > 0
    else 0
)
recall_similarity = (
    true_positive_similarity / (true_positive_similarity + false_negative_similarity)
    if (true_positive_similarity + false_negative_similarity) > 0
    else 0
)
fmeasure_similarity = (
    2 * precision_similarity * recall_similarity / (precision_similarity + recall_similarity)
    if (precision_similarity + recall_similarity) > 0
    else 0
)

# Calcola le metriche per il blocking basato sugli attributi
true_positive_attribute = attribute_match_count
false_positive_attribute = len(attribute_based_blocks) - true_positive_attribute
false_negative_attribute = len(groundtruth_df) - true_positive_attribute

precision_attribute = (
    true_positive_attribute / (true_positive_attribute + false_positive_attribute)
    if (true_positive_attribute + false_positive_attribute) > 0
    else 0
)
recall_attribute = (
    true_positive_attribute / (true_positive_attribute + false_negative_attribute)
    if (true_positive_attribute + false_negative_attribute) > 0
    else 0
)
fmeasure_attribute = (
    2 * precision_attribute * recall_attribute / (precision_attribute + recall_attribute)
    if (precision_attribute + recall_attribute) > 0
    else 0
)

# Stampa delle metriche calcolate manualmente
print("\nMetriche calcolate:")
print(f"Blocking basato sulla similarità: Precision: {precision_similarity:.2f}, Recall: {recall_similarity:.2f}, F-measure: {fmeasure_similarity:.2f}")
print(f"Blocking basato sugli attributi: Precision: {precision_attribute:.2f}, Recall: {recall_attribute:.2f}, F-measure: {fmeasure_attribute:.2f}")
