import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils import Prompter, get_logger

parser = argparse.ArgumentParser(prog="inference", description="Inference with Kullm")
parser.add_argument("--model-ckpt-path", type=str, help="Kullm model path")

def inference(args):
    logger = get_logger("inference")

    MODEL = args.model_ckpt_path
    logger.info(f'[+] Load Model from "{MODEL}"')

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device=f"cuda", non_blocking=True)
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

    prompter = Prompter("kullm")

    logger.info("[+] Load Dataset")
    data_path = "resource/data/nikluge-sc-2023-test.jsonl"
    test_data = load_dataset("json", data_files=data_path)
    test_data = test_data["train"]
    

    logger.info("[+] Start Inference")
    
    def infer(instruction="", input_text=""):
        prompt = prompter.generate_prompt(instruction, input_text)
        output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)

        return result




if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)