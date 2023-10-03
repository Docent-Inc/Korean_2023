import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPTNeoXTokenizerFast
import datetime
from src.utils import Prompter, get_logger
import warnings
from tqdm import tqdm
import os
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(prog="inference", description="Inference")
parser.add_argument("--geneartion-model-ckpt-path", type=str, default="outputs/generation",help="generation model path")
parser.add_argument("--validation-model-ckpt-path", type=str, default="outputs/validation",help="validation model path")
parser.add_argument("--adapter-model-ckpt-path", type=str, default="outputs/adapter",help="adapter model path")
parser.add_argument("--base_model", type=str, default="nlpai-lab/kullm-polyglot-5.8b-v2")

def infer(instruction="", input_text=""):
        prompt = prompter.generate_prompt(instruction, input_text)
        output = pipe(prompt, max_length=256, temperature=0.2, num_beams=3, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)

        return result

def merge_LoRA(BASE_MODEL, adapter, save_path):
    base_model = GPTNeoXForCausalLM.from_pretrained(
        BASE_MODEL,
        cache_dir="/media/mydrive",
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    embedding_size = base_model.get_input_embeddings().weight.size(1)
    # tokenizer = GPTNeoXTokenizerFast.from_pretrained(adapter)
    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    first_weight = base_model.gpt_neox.layers[0].attention.query_key_value.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        adapter,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    lora_weight = lora_model.base_model.gpt_neox.layers[0].attention.query_key_value.weight
    lora_model = lora_model.merge_and_unload()

    lora_model.train(False)
    lora_model_sd = lora_model.state_dict()
    deloreanized_sd = {k.replace("base_model.gpt_neox.", ""): v for k, v in lora_model_sd.items() if "lora" not in k}
    GPTNeoXForCausalLM.save_pretrained(base_model, save_directory=save_path, state_dict=deloreanized_sd)

def inference(args):
    logger = get_logger("inference")
    BASE_MODEL = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    for i in tqdm(range(1, k+1), desc="Processing folds"):
        geneartion_adapter = args.adapter_model_ckpt_path + f"/fold_generation_{i}"
        validation_adapter = args.adapter_model_ckpt_path + f"/fold_validation_{i}"

        # Marge_generation model
        MODEL = args.geneartion_model_ckpt_path
        logger.info(f'[+] Marge LoRA adapter {i} Generation Model from "{MODEL}"')
        merge_LoRA(BASE_MODEL, geneartion_adapter, MODEL)

        logger.info(f'[+] Load {i} Generation Model from "{MODEL}"')
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        ).to(device=f"cuda", non_blocking=True)
        
        model.eval()

        pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)
        prompter = Prompter("kullm")
        logger.info("[+] Load Dataset")
        data_path = "resource/data/nikluge-sc-2023-test.jsonl"
        test_data = load_dataset("json", data_files=data_path)
        test_data = test_data["train"]

        logger.info(f"[+] Start Generation {i}")
        result_file_name = f"output/files/generate_results_{i}.jsonl"
        special_token_id = 3  # <|sep|> 토큰
        special_token = tokenizer.decode([special_token_id])

        all_output_data_points = []  # 결과를 저장할 리스트

        for i, data_point in enumerate(tqdm(test_data, desc="Processing")):  # tqdm을 사용하여 진행 상태를 표시
            instruction = "문맥과 문법적 정확성 및 논리적 일관성에 맞는 자연스러운 한 문장이 되도록 두 문장 사이에 들어갈 한 문장을 접속사를 신경써서 만들어주세요."
            input_text = f"{data_point['input']['sentence1']} {special_token} {data_point['input']['sentence3']}"

            result = infer(instruction=instruction, input_text=input_text)

            output_data_point = {
                "id": data_point["id"],
                "input": data_point["input"],
                "output": result
            }

            all_output_data_points.append(output_data_point)  # 메모리에 결과 저장

        with open(result_file_name, "a") as f:
            for output_data_point in all_output_data_points:
                f.write(json.dumps(output_data_point, ensure_ascii=False) + "\n")

        logger.info(f"[+] Start Validation {i}")
        MODEL = args.validation_model_ckpt_path
        logger.info(f'[+] Marge LoRA adapter {i} Validation Model from "{MODEL}"')
        merge_LoRA(BASE_MODEL, geneartion_adapter, MODEL)

        logger.info(f'[+] Load {i} Validation Model from "{MODEL}"')
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        ).to(device=f"cuda", non_blocking=True)
        
        model.eval()

        pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)
        prompter = Prompter("kullm")
        logger.info("[+] Load Dataset")
        test_data = load_dataset("json", data_files=output_data_point)
        test_data = test_data["train"]

        result_file_name = f"output/files/validation_results_{i}.jsonl"
        all_output_data_points = []  # 결과를 저장할 리스트

        for i, data_point in enumerate(tqdm(test_data, desc="Processing")):  # tqdm을 사용하여 진행 상태를 표시
            instruction = "문맥과 문법적 정확성 및 논리적 일관성에 맞는 자연스러운 한 문장이 되도록 두 문장 이후에 나올 한 문장을 접속사를 신경써서 만들어주세요."
            input_text = f"{data_point['input']['sentence1']}{data_point['input']['output']}"

            result = infer(instruction=instruction, input_text=input_text)

            output_data_point = {
                "id": data_point["id"],
                "input": data_point["input"],
                "output": result
            }

            all_output_data_points.append(output_data_point)  # 메모리에 결과 저장

        with open(result_file_name, "a") as f:
            for output_data_point in all_output_data_points:
                f.write(json.dumps(output_data_point, ensure_ascii=False) + "\n")
    

if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)