import json

# Assuming the files are named file1.jsonl, file2.jsonl, ... file9.jsonl
file_names = [f"outputs/files/output{i}.jsonl" for i in range(1, 10)]

# Dictionary to store the counts
ensemble_counts = {}
data_store = {}

# Load each file and count the labels
for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            id_ = data['id']
            output = data['output']
            
            # Store the original data for later use
            if id_ not in data_store:
                data_store[id_] = data
            
            if id_ not in ensemble_counts:
                ensemble_counts[id_] = {label: 0 for label in output}
            
            for label, value in output.items():
                if value == "True":
                    ensemble_counts[id_][label] += 1

# Determine the final labels using hard voting and update the original data
threshold = len(file_names) // 2
for id_, counts in ensemble_counts.items():
    for label in data_store[id_]['output']:
        data_store[id_]['output'][label] = "True" if counts[label] > threshold else "False"

# Save the updated data to a new JSONL file
with open('outputs/ensembled_results.jsonl', 'w', encoding='utf-8') as f:
    for id_, data in data_store.items():
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
