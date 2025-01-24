import pandas as pd
from difflib import SequenceMatcher
from itertools import combinations

# Load the file
file_path = 'MEDIATED_SCHEMA/mediated_schema.csv'
data = pd.read_csv(file_path)

# Exclude the "Unified Attribute" column and work only on source columns
source_columns = data.columns[1:]  # All columns except the first

data_cleaned = data.iloc[3:, 1:].fillna('')  # Exclude header-like rows and the first column

# Define a function to calculate similarity based on unified attributes
def calculate_similarity_by_attributes(column1, column2):
    similar_attributes = 0
    total_attributes = 0  # Count of non-null elements in both columns

    for val1, val2 in zip(column1, column2):
        val1, val2 = val1.strip(), val2.strip()
        if val1 or val2:  # Consider only non-empty values
            total_attributes += 1
            if val1 and val2:  # Both values are non-empty
                similar_attributes += 1

    similarity_ratio = similar_attributes / total_attributes if total_attributes > 0 else 0
    return similarity_ratio, similar_attributes, total_attributes

# Generate similarity scores for all dataset column pairs
similarity_results = []

for col1, col2 in combinations(source_columns, 2):
    column1 = data_cleaned[col1].astype(str).fillna('')
    column2 = data_cleaned[col2].astype(str).fillna('')

    similarity_ratio, similar_attributes, total_attributes = calculate_similarity_by_attributes(column1, column2)

    similarity_results.append({
        "Dataset 1": col1,
        "Dataset 2": col2,
        "Similarity Ratio": similarity_ratio,
        "Similar Attributes": similar_attributes,
        "Total Attributes": total_attributes
    })

# Convert results to a DataFrame
similarity_df = pd.DataFrame(similarity_results)

# Sort by number of similar attributes and then by similarity ratio
similarity_df = similarity_df.sort_values(by=["Similar Attributes", "Similarity Ratio"], ascending=[False, False]).head(100)

# Save to a CSV file for review (overwrite mode)
output_path = 'GROUNDTRUTH/top_100_similar_datasets_by_attributes.csv'
similarity_df.to_csv(output_path, index=False, mode='w')

print(f"Top 100 similar datasets by attribute similarity saved to: {output_path}")
