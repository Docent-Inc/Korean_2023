from transformers import AutoTokenizer, GPTNeoXForCausalLM
import json
import numpy as np
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(prog="rejection_sampling", description="rejection_sampling")
parser.add_argument("--original-file-path", type=str, default="resource/data/nikluge-sc-2023-test.jsonl",help="original file path")
parser.add_argument("--base_model", type=str, default="nlpai-lab/kullm-polyglot-5.8b-v2")
parser.add_argument("--files-path", type=str, default="outputs/files",help="files path")
parser.add_argument("--k", type=int, default=5,help="k fold size")

def compute_similarity(sent1, sent2, model, tokenizer):
    vec1 = sentence_to_vector(sent1, model, tokenizer)
    vec2 = sentence_to_vector(sent2, model, tokenizer)
    return cosine_similarity(vec1, vec2)[0][0]

def sentence_to_vector(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state.mean(dim=1).numpy()

def inference(args):
    BASE_MODEL = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = GPTNeoXForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir="/media/mydrive",  
    )

    with open(args.original_file_path, 'r') as f:
        original_data = [json.loads(line) for line in f]
    id_to_sentence3 = {item['id']: item['input']['sentence3'] for item in original_data}

    ensemble_results = []

    for data in tqdm(original_data, desc="Processing data"):
        best_output = None
        best_similarity = -1
        data_id = data['id']
        sentence3 = id_to_sentence3[data_id]

        for i in range(1, args.k+1):
            file_path = args.files_path + f"/fold_validation_{i}.jsonl"
            with open(file_path, 'r') as f:
                fold_data = [json.loads(line) for line in f]
                fold_output = next(item for item in fold_data if item["id"] == data_id)["output"]
                similarity = compute_similarity(sentence3, fold_output, model, tokenizer)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_output = fold_output

        ensemble_results.append({
            "id": data_id,
            "input": data["input"],
            "output": best_output
        })

    with open(args.files_path + "/ensemble_results.jsonl", 'w') as f:
        for result in ensemble_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)