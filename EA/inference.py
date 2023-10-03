import json
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm

BATCH_SIZE = 512
labels = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]
tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

output_file_path = "predictions_5.jsonl"

model_number = {
    "joy" : 5,
    "anticipation" : 5,
    "trust" :5,
    "surprise" :5,
    "disgust" :5,
    "fear" :5,
    "anger" :5,
    "sadness" :5
}

# Load trained models
models = {}
for label in labels:
    model_path = f"outputs/model_{label}_{model_number[label]}"
    model = ElectraForSequenceClassification.from_pretrained(model_path).to(device)
    models[label] = model

def tokenize_data(texts):
    return tokenizer([text[0] for text in texts], [text[1] for text in texts], padding=True, truncation=True, max_length=256, return_tensors="pt")

def create_dataset(tokenized_data):
    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask']
    return TensorDataset(input_ids, attention_mask)

from tqdm import tqdm

def infer(data):
    texts = [(item['input']['form'], item['input']['target']['form']) for item in data]
    tokenized_data = tokenize_data(texts)
    dataset = create_dataset(tokenized_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    results = []
    idx = 0
    # Outer tqdm for batches
    for batch in tqdm(dataloader, desc="Processing batches", position=0, leave=True):
        batch_input_ids, batch_attention_mask = [b.to(device) for b in batch]
        
        # Inner tqdm for items within each batch
        for i in tqdm(range(len(batch_input_ids)), desc="Processing items", position=1, leave=False):
            item = data[idx]
            output = {}
            for label, model in models.items():
                with torch.no_grad():
                    logits = model(input_ids=batch_input_ids[i].unsqueeze(0), attention_mask=batch_attention_mask[i].unsqueeze(0)).logits
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    output[label] = "True" if pred[0] == 1 else "False"
            
            results.append({
                "id": item["id"],
                "input": item["input"],
                "output": output
            })
            idx += 1

    return results




if __name__ == "__main__":
    # Load test data
    with open("resource/data/splits/td_fold_6.jsonl", 'r') as f:    # "resource/data/nikluge-ea-2023-test.jsonl"
        test_data = [json.loads(line) for line in f]

    predictions = infer(test_data)
    
    print(predictions)

    # Save results
    with open(output_file_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"Inference completed and saved to {output_file_path}")





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

    def compute_f1(predictions, test_data):
        y_true = []
        y_pred = []
        
        for idx, item in enumerate(test_data):
            if "output" in item:  # Check if ground truth exists
                for label in labels:
                    y_true.append(1 if item["output"][label] == "True" else 0)
                    y_pred.append(1 if predictions[idx]["output"][label] == "True" else 0)
        
        if y_true:  # If ground truth was found
            f1 = calc_multi_label_classification_micro_F1(y_true, y_pred, average='micro')  # Compute micro F1 score
            return f1
        else:
            return None
        

    f1 = compute_f1(predictions, test_data)
    if f1 is not None:
        print(f"micro F1 Score: {f1:.4f}")
    else:
        print("No ground truth found in the test data.")