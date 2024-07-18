import os
os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
import sys
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")
import yaml
import torch
import json
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
from huggingface_hub import login

from lib.model_loader import load_model, trainer_loader
from lib.data_loader import data_loader_from_json
from utils.utils import CustomStream, load_config
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from lib.model_loader import load_model, load_model_eval, trainer_loader
from utils.utils import load_json

if __name__ == "__main__":
    config = load_config(file_name="./code/training/config/config.yml")
    new_model = './ckpt/fine_tuned_v3/checkpoint-80000/'
    model, tokenizer = load_model_eval(config, new_model)
    sysmsg = config.get("sysmsg")
    usermsg_prefix = config.get("usermsg_prefix")
    dataset_path = "./data/nacc_unique_with_llama_summaries_2_only_np.json"
    data = load_json(dataset_path)
    
    messages = [
        {"role": "system", "content": sysmsg},
        {"role": "user", "content": f"{usermsg_prefix} Don't generate Patient Medical History Summary. Just give the diagnosis. {data[200]['patient_LLAMA_SUMMARY']}."},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        # tokenize=False, 
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    outputs = model.generate(
            prompt,
            max_new_tokens=config.get("max_new_tokens"),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2,
        )
    response = outputs[0][prompt.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))