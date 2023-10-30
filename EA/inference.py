import json
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm

BATCH_SIZE = 512
labels = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]
tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_file_path = "predictions_5.jsonl"

model_number = {
    "joy" : 1,
    "anticipation" : 1,
    "trust" : 1,
    "surprise" : 1,
    "disgust" : 1,
    "fear" : 1,
    "anger" : 1,
    "sadness" : 1
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
    with open("resource/data/nikluge-ea-2023-test.jsonl", 'r') as f:    # "resource/data/nikluge-ea-2023-test.jsonl"
        test_data = [json.loads(line) for line in f]

    predictions = infer(test_data)
    
    print(predictions)

    # Save results
    with open(output_file_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"Inference completed and saved to {output_file_path}")