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


if __name__ == "__main__":
    config = load_config(file_name="./code/training/config/config.yml")

    model, tokenizer = load_model_eval(config)
    sysmsg = config.get("sysmsg")
    
    # messages = [
    #     {"role": "system", "content": "Using the information provided, your task is to act as a behavioral neurologist and provide a diagnostic summary for the patient described. Ensure your summary clearly identifies the patient's cognitive status and outlines the primary etiological diagnosis, if any cognitive disorders are detected."},
    #     {"role": "user", "content": "This 82-year-old female patient is a right-handed, widowed individual who lives alone. She has a history of hypertension, hypercholesterolemia, and thyroid disease. She is currently taking several medications, including ibuprofen, phenytoin, levothyroxine, omega-3 polyunsaturated fatty acids, simvastatin, and calcium-vitamin D.Neurological examination reveals a mild degree of slowness and poverty of movement, which is definitely abnormal. Her gait is normal, and she has no focal neurological signs or symptoms. She has a normal facial expression, and her eye movements are normal.Cognitive assessment using the Mini-Mental State Examination (MMSE) reveals a score of 24 out of 30. Her Boston Naming Test score is 9 out of 30, and her Digit Span Forward and Backward scores are 6 and 4, respectively.The patient's language skills are normal, and she is able to complete the Geriatric Depression Scale (GDS) without difficulty. She reports feeling happy most of the time and does not feel hopeless or helpless.The patient has a history of seizures, which are currently remote and inactive. She has no history of stroke, transient ischemic attack, or traumatic brain injury.In terms of functional abilities, the patient is able to perform daily activities such as preparing a balanced meal, paying attention to and understanding a TV program, and shopping alone for clothes and household necessities. However, she may require assistance with more complex tasks such as assembling tax records or business affairs. Her functional abilities are generally preserved, and she is able to live independently."}
    # ]
    
    messages = [
        {"role": "user", "content": "What is dementia?"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        # tokenize=False, 
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.float16,
    #     device_map=config.get("device_map"),
    # )

    # outputs = pipe(
    #     prompt, 
    #     max_new_tokens=config.get("max_new_tokens"), 
    #     do_sample=True, 
    #     temperature=0.6, 
    #     top_k=50, 
    #     top_p=0.7
    # )
    # print(outputs[0]["generated_text"])
    
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
            temperature=0.6,
            top_p=0.9,
        )
    response = outputs[0][prompt.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))