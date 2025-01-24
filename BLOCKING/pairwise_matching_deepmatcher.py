import pandas as pd

def prepare_deepmatcher_data(attribute_file, similarity_file, groundtruth_file):
    # Carica i file
    attribute_blocks = pd.read_csv(attribute_file)
    similarity_blocks = pd.read_csv(similarity_file)
    groundtruth = pd.read_csv(groundtruth_file)

    # Aggiungi etichette (label)
    def add_labels(blocks):
        blocks['label'] = blocks.apply(
            lambda row: 1 if ((row['Dataset 1'], row['Dataset 2']) in 
                              zip(groundtruth['Dataset 1'], groundtruth['Dataset 2'])) else 0,
            axis=1
        )
        return blocks

    attribute_blocks = add_labels(attribute_blocks)
    similarity_blocks = add_labels(similarity_blocks)

    # Aggiungi una colonna 'label' alla groundtruth
    groundtruth['label'] = 1  # Tutte le righe in groundtruth sono corrispondenze valide


    # Creare i file DeepMatcher
    def format_for_deepmatcher(blocks, output_file):
        blocks = blocks.rename(columns={
            'Dataset 1': 'left_Dataset',
            'Dataset 2': 'right_Dataset'
        })
        blocks['id'] = range(len(blocks))
        blocks[['id', 'left_Dataset', 'right_Dataset', 'label']].to_csv(output_file, index=False)

    # Salva i file per DeepMatcher
    format_for_deepmatcher(attribute_blocks, 'BLOCKING/data/train.csv')
    format_for_deepmatcher(similarity_blocks, 'BLOCKING/data/test.csv')
    format_for_deepmatcher(groundtruth, 'BLOCKING/data/validation.csv')

# Percorsi ai file
attribute_based_file = 'BLOCKING/attribute_based_blocks.csv'
similarity_based_file = 'BLOCKING/similarity_based_blocks.csv'
groundtruth_file = 'GROUNDTRUTH/top_100_similar_datasets_by_attributes.csv'

# Preparazione dei dati
prepare_deepmatcher_data(attribute_based_file, similarity_based_file, groundtruth_file)
