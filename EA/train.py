
import argparse
import json
import logging
import os
import sys

import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback
)
from datasets import Dataset, concatenate_datasets
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="klue/roberta-large", help="model file path")
g.add_argument("--tokenizer", type=str, default="klue/roberta-large", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=128, help="max sequence length")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--kfold", type=int, default=5, help="stratified K-fold")


def main(args):
    logger = logging.getLogger("train")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    
    
    # Load the initial datasets for Stratified K-fold
    logger.info(f'[+] Load Dataset')

    # Total number of folds
    num_folds = 9
    
    # If args.kfold is specified
    if 1 <= args.kfold <= num_folds:
        # Determine the validation fold based on args.kfold
        valid_file = f"resource/data/splits/td_fold_{args.kfold}.jsonl"
        valid_ds = Dataset.from_json(valid_file)
        
        # Load the other folds for training
        train_files = [f"resource/data/splits/td_fold_{i}.jsonl" for i in range(1, num_folds + 1) if i != args.kfold]
        train_datasets = [Dataset.from_json(file) for file in train_files]
        train_ds = concatenate_datasets(train_datasets)
    
    else:
        print("error: invalid K-fold N")
        exit()    
    

    labels = list(train_ds["output"][0].keys())
    id2label = {0: "False", 1: "True"}
    label2id = {"False": 0, "True": 1}

    for label in labels:
        logger.info(f"Training model for label: {label}")

        def preprocess_data_for_label(examples, label=label):
            text1 = examples["input"]["form"]
            text2 = examples["input"]["target"]["form"]
            encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=args.max_seq_len)
            encoding["labels"] = [label2id[examples["output"][label]]]
            return encoding

        encoded_tds = train_ds.map(preprocess_data_for_label, remove_columns=train_ds.column_names)
        encoded_vds = valid_ds.map(preprocess_data_for_label, remove_columns=valid_ds.column_names)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            problem_type="single_label_classification",
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
        
        targs = TrainingArguments(
            output_dir=os.path.join(args.output_dir, label),  # Save each model in a separate directory
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model= "accuracy",
        )


        def binary_metrics(predictions, labels):
            y_pred = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(y_true=labels, y_pred=y_pred)
            return {'accuracy': accuracy}

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            result = binary_metrics(predictions=preds, labels=p.label_ids)
            return result
        
        trainer = Trainer(
            model,
            targs,
            train_dataset=encoded_tds,
            eval_dataset=encoded_vds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()


if __name__ == "__main__":
    exit(main(parser.parse_args()))
