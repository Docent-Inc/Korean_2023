import os
import sys
from typing import List, Tuple
import warnings
import fire
import torch
from functools import partial
import multiprocessing
import transformers
from datasets import load_dataset
import torch.multiprocessing as mp
from tokenizers.processors import TemplateProcessing
from src.utils import Prompter, get_logger
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, EarlyStoppingCallback
warnings.filterwarnings("ignore", category=UserWarning)
mp.set_start_method('spawn', force=True)
_ = torch.cuda.FloatTensor(1)

logger = get_logger("train")

def tokenize(prompt, tokenizer, add_eos_token=True):
    cutoff_len = 256
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer, prompter, train_on_inputs, add_eos_token):
    special_token_id = 3  #  토큰
    special_token = tokenizer.decode([special_token_id])

    instruction = "문맥과 문법적 정확성 및 논리적 일관성에 맞는 자연스러운 한 문장이 되도록 두 문장 사이에 들어갈 한 문장을 접속사를 신경써서 만들어주세요."
    combined_input = f"{data_point['input']['sentence1']} {special_token} {data_point['input']['sentence3']}"

    full_prompt = prompter.generate_prompt(
        instruction,
        combined_input,
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(instruction, combined_input)
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def validate_and_tokenize_prompt(data_point, tokenizer, prompter, train_on_inputs):
    instruction = "두 문장 뒤에 이어질 자연스러운 한 문장을 만들어주세요."
    combined_input = f"{data_point['input']['sentence1']}{data_point['output']}"

    full_prompt = prompter.generate_prompt(
        instruction,
        combined_input,
        data_point['input']["sentence3"],
    )
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(instruction, combined_input)
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def init_model_and_tokenizer(base_model):
    model = GPTNeoXForCausalLM.from_pretrained(
        base_model,
        cache_dir="/media/mydrive",
        load_in_8bit=True,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model)
    return model, tokenizer

def train_generation(args: Tuple):
    i, base_model, tokenizer, train_data, val_data, micro_batch_size, gradient_accumulation_steps, num_epochs, learning_rate, ddp, group_by_length, use_wandb, wandb_run_name, val_set_size, output_dir, logger, lora_r, lora_alpha, lora_dropout, lora_target_modules = args
    model, tokenizer = init_model_and_tokenizer(base_model)
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    logger.info(f'[+] Train Generation Model')
    fold_output_dir = os.path.join(output_dir, f"fold_generation_{i}")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3, 
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(fold_output_dir)

def train_validation(args: Tuple):
    i, base_model, tokenizer, train_data, val_data, micro_batch_size, gradient_accumulation_steps, num_epochs, learning_rate, ddp, group_by_length, use_wandb, wandb_run_name, val_set_size, output_dir, logger, lora_r, lora_alpha, lora_dropout, lora_target_modules = args
    model, tokenizer = init_model_and_tokenizer(base_model)
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    logger.info(f'[+] Train Validation Model')
    fold_output_dir = os.path.join(output_dir, f"fold_validation_{i}")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
        model, type(model)
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(fold_output_dir)

def train(
    # model/data params
    base_model: str = "nlpai-lab/kullm-polyglot-5.8b-v2", # base mdoel 경로
    data_path: str = "resource/splits", # data_set 경로
    output_dir: str = "outputs/adapter", # output 경로
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 8,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    # k-fold Rejection Sampling
    k: int = 5,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["query_key_value", "xxx"],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Korean-AI",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "kullm",  # The prompt template to use, will default to alpaca.
):
    global model, tokenizer, prompter
    logger.info(f'[+] Save output to "{output_dir}"')

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    logger.info(f'[+] Load Tokenizer"')
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    for i in range(1, k+1):
        logger.info(f'[+] Load Dataset')
        fold_data_path = os.path.join(data_path, f"td_fold_{i}.jsonl")
        data = load_dataset("json", data_files=fold_data_path)["train"]
    
        # 데이터를 8:2 비율로 train과 validation으로 분할
        data = data.shuffle()
        train_size = int(0.8 * len(data))
        datasets_split = data.train_test_split(train_size=train_size, test_size=len(data)-train_size)
        train_data = datasets_split["train"]
        val_data = datasets_split["test"]
        val_set_size = len(val_data)

        # 토큰화 및 전처리
        partial_generate_and_tokenize = partial(generate_and_tokenize_prompt, tokenizer=tokenizer, prompter=prompter, train_on_inputs=train_on_inputs, add_eos_token=add_eos_token)
        train_data = train_data.shuffle().map(partial_generate_and_tokenize)
        partial_validate_and_tokenize = partial(validate_and_tokenize_prompt, tokenizer=tokenizer, prompter=prompter, train_on_inputs=train_on_inputs)
        val_data = val_data.shuffle().map(partial_validate_and_tokenize)

        args = (
            i, base_model, tokenizer, train_data, val_data, micro_batch_size,
            gradient_accumulation_steps, num_epochs, learning_rate, ddp, group_by_length,
            use_wandb, wandb_run_name, val_set_size, output_dir, logger, lora_r, lora_alpha, lora_dropout, lora_target_modules
        )

        # 병렬로 학습 실행
        process1 = mp.Process(target=train_generation, args=(args,))
        process2 = mp.Process(target=train_validation, args=(args,))
        process1.start()
        process2.start()
        process1.join()
        process2.join()
        break

        print("\n If there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":
    fire.Fire(train)