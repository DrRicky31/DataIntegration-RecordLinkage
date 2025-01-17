import pandas as pd

# Carica i dati
schema_file_path = '/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/MEDIATED_SCHEMA/unified_attribute_mapping.csv'
groundtruth_file_path = '/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/GROUNDTRUTH/top_100_similar_datasets_by_attributes.csv'
schema_df = pd.read_csv(schema_file_path)
groundtruth_df = pd.read_csv(groundtruth_file_path)

# Strategia 1: Attribute-Based Blocking
attribute_threshold = 2
attribute_based_blocks = groundtruth_df[
    groundtruth_df["Similar Attributes"] >= attribute_threshold
]

# Salva il risultato della strategia 1 in un file CSV
attribute_based_blocks.to_csv('/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/BLOCKING/attribute_based_blocks.csv', index=False)

# Strategia 2: Similarity Ratio Threshold Blocking
similarity_threshold = 0.75
similarity_based_blocks = groundtruth_df[
    groundtruth_df["Similarity Ratio"] >= similarity_threshold
]

# Salva il risultato della strategia 2 in un file CSV
similarity_based_blocks.to_csv('/home/alessio/Desktop/UNI MAGISTRALE/ESAMI DA FARE/INGEGNERIA DEI DATI/Homework/Hw5_IDD/Hw5_IDD/BLOCKING/similarity_based_blocks.csv', index=False)