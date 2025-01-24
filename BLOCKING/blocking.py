import pandas as pd

# Carica i dati
schema_file_path = 'MEDIATED_SCHEMA/mediated_schema.csv'
groundtruth_file_path = 'GROUNDTRUTH/top_100_similar_datasets_by_attributes.csv'
schema_df = pd.read_csv(schema_file_path)
groundtruth_df = pd.read_csv(groundtruth_file_path)

# Strategia 1: Attribute-Based Blocking
attribute_threshold = 2
attribute_based_blocks = groundtruth_df[
    groundtruth_df["Similar Attributes"] >= attribute_threshold
]

# Salva il risultato della strategia 1 in un file CSV
attribute_based_blocks.to_csv('BLOCKING/attribute_based_blocks.csv', index=False)

# Strategia 2: Similarity Ratio Threshold Blocking
similarity_threshold = 0.15
similarity_based_blocks = groundtruth_df[
    groundtruth_df["Similarity Ratio"] >= similarity_threshold
]

# Salva il risultato della strategia 2 in un file CSV
similarity_based_blocks.to_csv('BLOCKING/similarity_based_blocks.csv', index=False)