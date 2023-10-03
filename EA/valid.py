from sklearn.metrics import f1_score

def calc_multi_label_classification_micro_F1(true, pred):

    if type(true[0]) is list:
        if type(true[0][0]) is int:
            pass
        elif type(true[0][0]) is float:
            pass
        elif type(true[0][0]) is bool:
            pass
        elif type(true[0][0]) is str:
            pass
        else:
            return -1

    elif type(true[0]) is dict:

        sample_key = next(iter(true[0]))

        if type(true[0][sample_key]) is int:
            pass
        elif type(true[0][sample_key]) is float:
            pass
        elif type(true[0][sample_key]) is str:
            def dict_to_list(input_dict):
                output_list = []
                for instance in input_dict.values():
                    if instance == 'True' or instance == 'true':
                        output_list.append(1)
                    else:
                        output_list.append(0)

                return output_list

            formated_pred = list(map(lambda x: dict_to_list(x), pred))
            formated_true = list(map(lambda x: dict_to_list(x), true))
            f1_micro = f1_score(y_true=formated_true, y_pred=formated_pred, average='micro')

            return f1_micro

        elif type(true[0][sample_key]) is bool:
            def dict_to_list(input_dict):
                output_list = []
                for instance in input_dict.values():
                    if instance is True:
                        output_list.append(1)
                    else:
                        output_list.append(0)

            formated_pred = list(map(lambda x: dict_to_list(x), pred))
            formated_true = list(map(lambda x: dict_to_list(x), true))
            f1_micro = f1_score(y_true=formated_true, y_pred=formated_pred, average='micro')
            return f1_micro

        else:
            return -1
    else:
        return -1
    

import json

with open("resource/data/splits/td_fold_5.jsonl", 'r') as f:    # "resource/data/nikluge-ea-2023-test.jsonl"
    true_label = [json.loads(line) for line in f]
    
with open("predictions_5.jsonl", 'r') as f:    # "resource/data/nikluge-ea-2023-test.jsonl"
    predicted_label = [json.loads(line) for line in f]
    
# Extracting the 'output' key from the labels
true_output = [sample['output'] for sample in true_label]
predicted_output = [sample['output'] for sample in predicted_label]

print(calc_multi_label_classification_micro_F1(true_output, predicted_output))





# 1. Identify the indices of the wrong predictions.
wrong_indices = [i for i, (true, pred) in enumerate(zip(true_output, predicted_output)) if true != pred]

# 2. Extract the corresponding samples from the true_label and predicted_label lists.
wrong_true_labels = [true_label[i] for i in wrong_indices]
wrong_predicted_labels = [predicted_label[i] for i in wrong_indices]

# 3. Save the wrong predictions to a file.
with open("wrong_predictions.jsonl", 'w') as f:
    for true, pred in zip(wrong_true_labels, wrong_predicted_labels):
        f.write(json.dumps({"true": true, "predicted": pred['output']}, ensure_ascii=False) + '\n')






from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_single_label(true, pred, label_key):
    """
    Evaluate precision, recall, and F1 score for a single label.
    
    Parameters:
    - true: List of true multi-label samples.
    - pred: List of predicted multi-label samples.
    - label_key: Key of the label for which metrics are to be calculated.
    
    Returns:
    - precision, recall, f1 for the specified label.
    """
    
    # Extract the specific label from each sample
    true_single_label = [sample[label_key] for sample in true]
    pred_single_label = [sample[label_key] for sample in pred]
    
    
    # Convert boolean or string labels to binary (0 or 1)
    true_single_label = [1 if label == "True" else 0 for label in true_single_label]
    pred_single_label = [1 if label == "True" else 0 for label in pred_single_label]
    
    # Calculate precision, recall, and F1 score
    precision = precision_score(true_single_label, pred_single_label)
    recall = recall_score(true_single_label, pred_single_label)
    f1 = f1_score(true_single_label, pred_single_label)
    
    return precision, recall, f1

# Example usage:
label_key = "joy"  # Change this to the key of the label you want to evaluate
precision, recall, f1 = evaluate_single_label(true_output, predicted_output, label_key)
print(f"Precision for label {label_key}: {precision}")
print(f"Recall for label {label_key}: {recall}")
print(f"F1 Score for label {label_key}: {f1}")