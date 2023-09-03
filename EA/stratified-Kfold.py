import numpy as np
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import StratifiedKFold

# Load the datasets
data1 = Dataset.from_json("resource/data/nikluge-ea-2023-train.jsonl")
data2 = Dataset.from_json("resource/data/nikluge-ea-2023-dev.jsonl")

# Concatenate the datasets
combined_data = concatenate_datasets([data1, data2])

# Label distribution and their priority (rarer labels have higher priority)
label_priority = {
    'fear': 1,
    'disgust': 2,
    'anger': 3,
    'trust': 4,
    'sadness': 5,
    'surprise': 6,
    'anticipation': 7,
    'joy': 8
}

# Assuming 'output' in your dataset is a dictionary with labels as keys and values as "True"/"False"
# Convert this to a priority label for each instance
priority_labels = []
for entry in combined_data:
    labels = [label for label, value in entry['output'].items() if value == "True"]
    if labels:
        # Get the label with the highest priority (lowest number)
        priority_label = min(labels, key=lambda x: label_priority[x])
        priority_labels.append(label_priority[priority_label])
    else:
        # Handle instances with no labels (you can assign a default priority or handle it differently)
        priority_labels.append(0)

# Perform stratified K-fold on the priority labels
skf = StratifiedKFold(n_splits=9)
for train_index, test_index in skf.split(np.zeros(len(priority_labels)), priority_labels):
    print("Train:", train_index)
    print("Test:", test_index)
    
    

import os

# Directory to save the test JSON files
output_dir = "resource/data/splits"

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Iterate over the folds and save the test sets
for fold_num, (_, test_index) in enumerate(skf.split(np.zeros(len(priority_labels)), priority_labels), 1):
    # Subset the data using test indices
    test_subset = combined_data.select(test_index)
    
    # Convert to pandas DataFrame
    df = test_subset.to_pandas()
    
    # Save the subset as a JSONL file
    output_file = os.path.join(output_dir, f"td_fold_{fold_num}.jsonl")
    
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    
    print(f"Saved test set for fold {fold_num} to {output_file}")
