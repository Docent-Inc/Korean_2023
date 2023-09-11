
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(prog="train", description="Inference Table to Text with BART")

parser.add_argument("--model-ckpt-path", type=str, help="model path")
parser.add_argument("--output-path", type=str, default="outputs/files/output2.jsonl", help="output tsv file path")
parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
parser.add_argument("--max-seq-len", type=int, default=512, help="summary max sequence length")
parser.add_argument("--threshold", type=float, default=0.5, help="inferrence threshold")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


def main(args):
    logger = logging.getLogger("inference")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    logger.info(f"[+] Use Device: {args.device}")
    device = torch.device(args.device)

    logger.info(f'[+] Load Tokenizer from "{args.model_ckpt_path}"')
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt_path)

    logger.info(f'[+] Load Dataset')
    test_ds = Dataset.from_json("resource/data/nikluge-ea-2023-test.jsonl")
    with open(os.path.join(args.model_ckpt_path, "..", "label2id.json")) as f:
        label2id = json.load(f)
    labels = list(label2id.keys())
    id2label = {}
    for k, v in label2id.items():
        id2label[v] = k

    def preprocess_data(examples):
        # take a batch of texts
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=args.max_seq_len)

        return encoding

    encoded_tds = test_ds.map(preprocess_data, remove_columns=test_ds.column_names).with_format("torch")
    data_loader = DataLoader(encoded_tds, batch_size=args.batch_size)

    logger.info("[+] Eval mode & Disable gradient")
    torch.set_grad_enabled(False)

    sigmoid = torch.nn.Sigmoid()
    outputs = [np.zeros((len(test_ds), 1)) for _ in labels]  # Initialize outputs

    for label_idx, label in enumerate(labels):
        logger.info(f'[+] Load Model for label: {label} from "{os.path.join(args.model_ckpt_path, label)}"')
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(args.model_ckpt_path, label),  # Load model from label-specific directory
            problem_type="single_label_classification",
            num_labels=2,
            id2label={0: "False", 1: "True"},
            label2id={"False": 0, "True": 1}
        )
        model.to(device)
        model.eval()

        logger.info(f"[+] Start Inference for label: {label}")
        for batch in tqdm(data_loader):
            oup = model(
                batch["input_ids"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            probs = sigmoid(oup.logits).cpu().detach().numpy()
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs[:, 1] >= args.threshold)] = 1  # Use the second column (index 1) for "True" label
            outputs[label_idx] = y_pred

    def jsonlload(fname):
        with open(fname, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
            j_list = [json.loads(line) for line in lines]

        return j_list

    def jsonldump(j_list, fname):
        with open(fname, "w", encoding='utf-8') as f:
            for json_data in j_list:
                f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

    j_list = jsonlload("resource/data/nikluge-ea-2023-test.jsonl")
    for idx, _ in enumerate(j_list):
        j_list[idx]["output"] = {}
        for label_idx, label in enumerate(labels):
            if outputs[label_idx][idx]:
                j_list[idx]["output"][label] = "True"
            else:
                j_list[idx]["output"][label] = "False"

    jsonldump(j_list, args.output_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
