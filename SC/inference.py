import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.utils import Prompter, get_logger

parser = argparse.ArgumentParser(prog="inference", description="Inference with Kullm")
parser.add_argument("--model-ckpt-path", type=str, default="nlpai-lab/kullm-polyglot-5.8b-v2",help="Kullm model path")

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

    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    result_file_name = f"inference_results_{current_time}.jsonl"
    special_token_id = 3  # <|sep|> 토큰
    special_token = tokenizer.decode([special_token_id])

    with open(result_file_name, "w") as f:
        for data_point in test_data:
            instruction = "문맥에 맞는 자연스러운 한 문장이 되도록 두 문장 사이에 들어갈 한 문장을 만들어주세요."
            input_text = f"{data_point['input']['sentence1']} {special_token} {data_point['input']['sentence3']}"

            # 추론을 수행합니다.
            result = infer(instruction=instruction, input_text=input_text에)

            output_data_point = {
                "id": data_point["id"],
                "input": data_point["input"],
                "output": result
            }
            f.write(json.dumps(output_data_point) + "\n")

if __name__ == "__main__":
    args = parser.parse_args()
    inference(args)