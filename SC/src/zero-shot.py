import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils import Prompter

MODEL = "nlpai-lab/kullm-polyglot-5.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    print(prompt)

    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result



result = infer(input_text="""우진이는 폭우로 독서실이 침수되었다는 소식을 들었다.

다행히 우진이가 다니는 독서실은 무사해서 우진이는 안도했다.

이를 통해 추론할 수 있는 숨겨진 내용은?(적혀있지 않은 내용만, 평서문)""")

print(result)