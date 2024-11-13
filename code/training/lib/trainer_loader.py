# coding=utf-8
#
# LICENSE OF THE FOLLOWING MODELS
#
# LLAMA 3 COMMUNITY LICENSE AGREEMENT
# https://llama.meta.com/llama3/license/

import gc
import torch
import wandb
import os
import torch.nn as nn
import trl

from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from trl.trainer import SFTConfig
from utils.utils import print_parameters
from huggingface_hub import login

def load_trainer(config, model, tokenizer, dataset, num_train_epochs):
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
    hf_write_token = config.get("hf_write_token")
    push_to_hub = config.get("push_to_hub")
    hub_model_id = config.get("hub_model_id")

    arguments = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_train_epochs,
        split_batches=True,
        per_device_train_batch_size=train_batch_size,
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
        hub_token=hf_write_token,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_private_repo=True,
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
        dataset_text_field="text",
        packing=False,
        args=arguments,
    )

    return trainer