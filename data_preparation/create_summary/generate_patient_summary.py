import os
os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import numpy as np
import torch.distributed as dist
import json
import warnings
import random
import time
import string
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import timedelta
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
os.environ['VLLM_SKIP_P2P_CHECK'] = "1"
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

PATIENT_SUMMARY_PROMPT = """You will receive patient data between <data> and </data> tags. Summarize the patient information provided without making any assumptions or conclusions. Include important neuropsychological battery summary scores whenever availale. Do not use bullet points, numbered lists, or section headings; craft coherent paragraphs and transition naturally between topics. Write in continuous prose using complete sentences.

<data>
{patient}
</data>
"""

parser = argparse.ArgumentParser(description="Process model parameters.")
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
parser.add_argument('--max_new_tokens', type=int, default=10000, help='max_new_tokens')
parser.add_argument('--max_model_len', type=int, default=10000, help='Maximum model length')
parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Model ID')
parser.add_argument('--n_devices', type=int, default=4, help='Number of devices')
parser.add_argument('--start_id', type=int, default=0, help='Start ID')
parser.add_argument('--end_id', type=int, default=10000000, help='End ID')
parser.add_argument('--directory_path', type=str, default="/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/pretrain_summaries", help='Directory path')
parser.add_argument('--data_path', type=str, default="/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/pretrain_summaries/step_1/qwen72B_step_1_0_47408.csv", help='Data path')
parser.add_argument('--json_name', type=str, default="qwen72B_step_2_0_10000.csv", help='JSON output file name')

# Parse arguments
args = parser.parse_args()
json_path = f"{args.directory_path}/{args.json_name}"

def load_model():
    """Load VLLM model and Huggingface tokenizer."""
    print(args.model_id)
    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=args.n_devices,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        enable_lora=False,
        distributed_executor_backend='mp',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return llm, tokenizer


def get_vllm_summary(llm, tokenizer, messages, max_new_tokens=3000):
    """This is a function to generate LLAMA summaries using vllm https://github.com/vllm-project/vllm

    Args:
        llm: LLM object
        tokenizer: Huggingface tokenizer
        input_texts (List): A list of input texts / prompts
        system_msg (str): system message for the LLAMA prompt

    Returns:
        List: A list of generated responses
    """
    

    prompts = []
    responses = []
    
    for message in messages:
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
            continue_final_message=False,
            enable_thinking=False
            # return_tensors="pt"
        )
        
        prompts.append(input_ids)
    
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        max_tokens=max_new_tokens,
        presence_penalty=0.2,
        # frequency_penalty=0.5,
        # stop=stop_tokens
    )
    
    completions = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    
    for i, output in enumerate(completions):
        temp_gen = output.outputs[0].text
        responses.append(temp_gen)
        
    # print('Successfully finished generating', len(prompts), 'samples!')
    
    return responses

def create_batches(dataframe):
    """Yield successive n-sized batches from dataframe."""
    for i in range(0, len(dataframe), args.batch_size):
        yield dataframe.iloc[i:i + args.batch_size]


def generate_summary(patient_files, llm, tokenizer, max_new_tokens=2048):
    """
    Generate summaries for patient data using a language model.
    
    Args:
    - patient_files: List of strings containing JSON-encoded patient data.
    - system_msg: Initial system message for the language model.
    - llm, tokenizer: Language model and tokenizer for generating summaries.

    Returns:
    - List of patient summaries.
    """
    messages = []
    for patient_file in patient_files:
        patient_file_json = json.loads(patient_file)
        if 'Co-participant Demographics' in patient_file_json:
            del patient_file_json['Co-participant Demographics']
            
        if "Subject's month of birth" in patient_file_json['Subject Demographics']:
            del patient_file_json['Subject Demographics']["Subject's month of birth"]
            
        if "Subject's year of birth" in patient_file_json['Subject Demographics']:
            del patient_file_json['Subject Demographics']["Subject's year of birth"]
            
        if "Subject's age at visit" in patient_file_json['Subject Demographics']:
            key_to_move_first = "Subject's age at visit"
            patient_file_json['Subject Demographics'] = {key_to_move_first: patient_file_json['Subject Demographics'][key_to_move_first], **{k: v for k, v in patient_file_json['Subject Demographics'].items() if k != key_to_move_first}}
            
        patient_file = json.dumps(patient_file_json, indent=4)
            
        message = [
            # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": PATIENT_SUMMARY_PROMPT.format(patient=patient_file)}
        ]
        messages.append(message)
        
    patient_summaries = get_vllm_summary(llm, tokenizer, messages, max_new_tokens=max_new_tokens)
    
    return patient_summaries
            
if __name__ == "__main__":
    print(f"Using model {args.model_id}")
    # Loading model
    llm, tokenizer = load_model()
    
    # Loading data
    print(f"Loading data: {args.data_path}")
    train_data = pd.read_csv(args.data_path)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)[args.start_id : min(args.end_id, len(train_data))]
    
    if not os.path.exists(f"{args.directory_path}"):
        os.makedirs(f"{args.directory_path}", exist_ok=True)
    
    print(f"Generating summaries for {len(train_data)} cases.")
    for batch in tqdm(create_batches(train_data)):
        query_texts = batch['patient_summary'].tolist()
        
        generated_summaries = generate_summary(query_texts, llm, tokenizer, max_new_tokens=args.max_new_tokens)
        print('Successfully finished generating', len(generated_summaries), 'samples!')
                
        data_to_save = []
        for i, row in batch.iterrows():
            summary_data = dict(row)
            summary_data['visit_summary'] =  generated_summaries[i - batch.first_valid_index()]
            data_to_save.append(summary_data)

        # Convert list of dictionaries to DataFrame
        df_to_save = pd.DataFrame(data_to_save)

        # Append DataFrame to CSV, creating the file if it does not exist yet
        df_to_save.to_csv(json_path, mode='a', index=False, header=not os.path.exists(json_path), encoding='utf-8')
    
    # Destroying vllm instance
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
        
        