# coding=utf-8
#
# LICENSE OF THE FOLLOWING MODELS
#
# LLAMA 3 COMMUNITY LICENSE AGREEMENT
# https://llama.meta.com/llama3/license/

import gc
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
import trl
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from trl.trainer import SFTConfig
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from utils.utils import print_parameters


def load_model(config, load_from_adaptor=False, new_model=None):
    """
    Initialize model
    :param config: the YAML configuration file
    """
    model_name = config.get("model_name")
    device_map = config.get("device_map")
    hf_read_token = config.get("hf_read_token")
    save_dir = config.get("save_dir")
    lora_r = config.get("lora_r")
    lora_alpha = config.get("lora_alpha")
    lora_dropout = config.get("lora_dropout")
    cache_dir = config.get("cache_dir")

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_read_token,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding='longest',
        padding_side="right",
        truncation=True,
        return_tensors="pt",
        use_auth_token=hf_read_token,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    
    if "llama" in model_name.lower() or "mistralai" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    
    # model, tokenizer = setup_chat_format(model, tokenizer)

    if not load_from_adaptor:   
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
            ],
            task_type=TaskType.CAUSAL_LM,
        )
        # model = get_peft_model(model, peft_config, adapter_name="adapter")
        model.add_adapter(peft_config, adapter_name="adapter")
        model.enable_adapters()
        print_parameters(model=model)

        # Save the tokenizer
        tokenizer.save_pretrained(save_dir)
        print('Successfully save the tokenizer!')

        # Save the pre-trained model
        model.save_pretrained(save_dir)
        print('Successfully save the model!\n\n')

    else:
        print("Loading from adaptor")
        
        # Merge adapter with base model
        model = PeftModel.from_pretrained(model, new_model, adapter_name="adapter", torch_device='cpu')
        model = model.merge_and_unload()

    # Clean the cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, tokenizer


def load_model_eval(config, new_model):
    base_model = config.get("model_name")
    device_map = config.get("device_map")
    hf_read_token = config.get("hf_read_token")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_auth_token=hf_read_token,
    )

    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=hf_read_token,
    )


    if "llama" in base_model.lower() or "mistralai" in base_model.lower():
        tokenizer.pad_token = tokenizer.eos_token

    # Merge adapter with base model
    model = PeftModel.from_pretrained(base_model_reload, new_model)
    model = model.merge_and_unload()
    return model, tokenizer
    # return base_model_reload, tokenizer
    

def trainer_loader(config, model, tokenizer, dataset, num_train_epochs):
    """
    Load training pipeline
    :param config: the configurations
    :param model: the pre-trained model
    :param tokenizer: the tokenizer of the pre-trained model
    :param dataset: the training dataset
    :param num_train_epochs: the number of training epochs
    :return trainer: SFTTrainer
    """
    model_name = config.get("model_name")
    train_batch_size = config.get("train_batch_size")
    gradient_accumulation_steps = config.get("gradient_accumulation_steps")
    optim = config.get("optim")
    logging_steps = config.get("logging_steps")
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    warmup_ratio = config.get("warmup_ratio")
    lr_scheduler_type = config.get("lr_scheduler_type")
    fp16 = config.get("fp16")
    bf16 = config.get("bf16")
    save_dir = config.get("save_dir")
    train_max_len = config.get("train_max_len")
    gradient_checkpointing = config.get("gradient_checkpointing")
    log_save_platform = config.get("log_save_platform")
    save_strategy = config.get("save_strategy")
    save_steps = config.get("save_steps")
    save_only_model = config.get("save_only_model")
    save_total_limit = config.get("save_total_limit")


    arguments = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_train_epochs,
        split_batches=True,
        per_device_train_batch_size=train_batch_size,
        # per_device_eval_batch_size=train_batch_size,
        # auto_find_batch_size=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        warmup_ratio=warmup_ratio,
        report_to=log_save_platform,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_only_model=save_only_model,
        save_total_limit=save_total_limit,
    )

    # instruction_template = tokenizer.encode("user", add_special_tokens=False)
    # response_template = tokenizer.encode("assistant", add_special_tokens=False)
    instruction_template = "### Instruction:\n"
    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        # eval_dataset=dataset['val'],
        tokenizer=tokenizer,
        data_collator=collator,
        max_seq_length=train_max_len,
        # max_seq_length=tokenizer.model_max_length,
        dataset_text_field="text",
        packing=False,
        args=arguments,
    )

    return trainer
