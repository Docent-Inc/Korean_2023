from datasets import Dataset, concatenate_datasets

# # Load the datasets
# data1 = Dataset.from_json("resource/data/splits/test_fold_1.jsonl")
# data2 = Dataset.from_json("resource/data/splits/test_fold_2.jsonl")
# data3 = Dataset.from_json("resource/data/splits/test_fold_3.jsonl")
# data4 = Dataset.from_json("resource/data/splits/test_fold_4.jsonl")
# data5 = Dataset.from_json("resource/data/splits/test_fold_5.jsonl")

# # Concatenate the datasets
# combined_data = concatenate_datasets([data1, data2, data3, data4, data5])


data1 = Dataset.from_json("resource/data/nikluge-ea-2023-train.jsonl")
data2 = Dataset.from_json("resource/data/nikluge-ea-2023-dev.jsonl")
combined_data = concatenate_datasets([data1, data2])

print(len(combined_data))


combined_data = Dataset.from_json("resource/data/nikluge-ea-2023-test.jsonl")

# # Initialize a dictionary to count the label distribution
# label_counts = {
#     "joy": 0,
#     "anticipation": 0,
#     "trust": 0,
#     "surprise": 0,
#     "disgust": 0,
#     "fear": 0,
#     "anger": 0,
#     "sadness": 0
# }

# # Count the labels
# for entry in combined_data:
#     for label, value in entry["output"].items():
#         if value == "True":
#             label_counts[label] += 1

# # Print the label distribution
# for label, count in label_counts.items():
#     print(f"{label}: {count}")

# # If you have matplotlib installed, you can visualize the distribution
# try:
#     import matplotlib.pyplot as plt

#     labels = list(label_counts.keys())
#     counts = list(label_counts.values())

#     plt.bar(labels, counts)
#     plt.ylabel('Counts')
#     plt.title('Label Distribution')
#     plt.xticks(rotation=45)
#     plt.show()
# except ImportError:
#     print("matplotlib not installed. Consider installing it for visualization.")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('beomi/KcELECTRA-base-v2022')

# Tokenize the data and compute the lengths
token_lengths = [len(tokenizer.encode(item['input']['form'], item['input']['target']['form'])) for item in combined_data]  # Replace 'input' with the appropriate key if different

# Print the maximum token length
print(f"Maximum token length: {max(token_lengths)}")