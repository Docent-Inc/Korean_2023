import json
from transformers import ElectraTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from transformers import ElectraForSequenceClassification, AdamW
from sklearn.metrics import f1_score
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import wandb
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default="beomi/KcELECTRA-base-v2022")
parser.add_argument('--seed' , type=int , default = 1, help='random seed (default: 1)')
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=9999)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--early_stop_patient', type=int, default=5)
parser.add_argument('--nsplit', type=int, default=9, help='n split K-Fold')
parser.add_argument('--kfold', type=int, default=1, help='n split K-Fold')
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
parser.add_argument('--wandb', type=int, default=1, help='wandb on / off')

args = parser.parse_args()


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def set_seed(seedNum):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(args.seed)


# If args.kfold is specified
if 1 <= args.kfold <= args.nsplit:
    # Determine the validation fold based on args.kfold
    dev_file = f"resource/data/splits/td_fold_{args.kfold}.jsonl"
    dev_data = Dataset.from_json(dev_file)

    # Load the other folds for training
    train_files = [f"resource/data/splits/td_fold_{i}.jsonl" for i in range(1, args.nsplit + 1) if i != args.kfold]
    train_datasets = [Dataset.from_json(file) for file in train_files]
    train_data = concatenate_datasets(train_datasets)

else:
    print("error: invalid K-fold N")
    exit()
        
        

labels = ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]
# ["joy", "anticipation", "trust", "surprise", "disgust", "fear", "anger", "sadness"]

train_texts = [(item['input']['form'], item['input']['target']['form']) for item in train_data]
train_labels = [[int(item['output'][label] == "True") for label in labels] for item in train_data]

dev_texts = [(item['input']['form'], item['input']['target']['form']) for item in dev_data]
dev_labels = [[int(item['output'][label] == "True") for label in labels] for item in dev_data]





tokenizer = ElectraTokenizer.from_pretrained(args.model_name)

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


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)



if args.wandb:
    config = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "early_stop_patient" : args.early_stop_patient,
        "wandb" : args.wandb,
        "K-Fold" : f'{args.kfold}/{args.nsplit}'
    }
    
    wandb_name = f'{args.kfold}/{args.nsplit}_{args.lr}_{args.batch_size}'

    wandb.init(entity="docent-research", project="EA", name = wandb_name, config = config)



for idx, label in enumerate(labels):
    print(f"Training model for label: {label}")

    model = ElectraForSequenceClassification.from_pretrained(args.model_name, num_labels=2).to(device)
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        entity_property_param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        entity_property_optimizer_grouped_parameters = [
            {'params': [p for n, p in entity_property_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': args.weight_decay},
            {'params': [p for n, p in entity_property_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
        ]
    optimizer = AdamW(entity_property_optimizer_grouped_parameters, lr=args.lr)

    early_stop_counter = 0
    best_f1 = 0

    for epoch in range(args.epochs):  # Assuming max 10 epochs, you can adjust
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
                
        f1 = f1_score(all_true, all_preds, average = "macro")
        
        print(f"Epoch {epoch + 1} | Eval F1 score: {f1}")

        if f1 > best_f1:
            print("Best epoch ! saved the model")
            best_f1 = f1
            early_stop_counter = 0
            model.save_pretrained(f"outputs/model_{label}_{args.kfold}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= args.early_stop_patient:
            break

    print(f"Best F1 for {label}: {best_f1}")
    wandb.log({label : best_f1})
