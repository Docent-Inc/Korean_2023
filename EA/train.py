import json
from transformers import ElectraTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from transformers import ElectraForSequenceClassification, AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

train_data = load_data("resource/data/nikluge-ea-2023-train.jsonl")
dev_data = load_data("resource/data/nikluge-ea-2023-dev.jsonl")

labels = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]

train_texts = [(item['input']['form'], item['input']['target']['form']) for item in train_data]
train_labels = [[int(item['output'][label] == "True") for label in labels] for item in train_data]

dev_texts = [(item['input']['form'], item['input']['target']['form']) for item in dev_data]
dev_labels = [[int(item['output'][label] == "True") for label in labels] for item in dev_data]







BATCH_SIZE = 64

tokenizer = ElectraTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

def tokenize_data(texts):
    return tokenizer([text[0] for text in texts], [text[1] for text in texts], padding=True, truncation=True, return_tensors="pt")

def create_dataset(tokenized_data, labels):
    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask']
    return TensorDataset(input_ids, attention_mask, torch.tensor(labels))

tokenized_train = tokenize_data(train_texts)
train_dataset = create_dataset(tokenized_train, train_labels)

tokenized_dev = tokenize_data(dev_texts)
dev_dataset = create_dataset(tokenized_dev, dev_labels)


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, label in enumerate(labels):
    print(f"Training model for label: {label}")

    model = ElectraForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022", num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    early_stop_counter = 0
    best_f1 = 0

    for epoch in range(100):  # Assuming max 10 epochs, you can adjust
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
            current_batch_labels = batch_labels[:, idx]

            optimizer.zero_grad()
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=current_batch_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()


        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_dataloader)}")

        # Evaluation
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch in dev_dataloader:
                batch_input_ids, batch_attention_mask, batch_labels = [b.to(device) for b in batch]
                current_batch_labels = batch_labels[:, idx].cpu().numpy()

                outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_true.extend(current_batch_labels)
                

        f1 = f1_score(all_true, all_preds)
        
        print(f"Epoch {epoch + 1} | Eval F1 score: {f1}")

        if f1 > best_f1:
            print("Best epoch ! saved the model")
            best_f1 = f1
            early_stop_counter = 0
            model.save_pretrained(f"outputs/model_{label}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= 3:
            break

    print(f"Best F1 for {label}: {best_f1}")
