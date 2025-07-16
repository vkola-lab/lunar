# generate_patient_summary.py

# Created by: Sahana Kowshik
# Description: Script to create visit summaries from patient json files using VLLM


import os
os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
os.environ['VLLM_SKIP_P2P_CHECK'] = "1"
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import numpy as np
import torch.distributed as dist
import json
import warnings
warnings.filterwarnings("ignore")
import random
import time
import string
import gc
import ray

from tqdm import tqdm
from datetime import timedelta
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

# PATIENT_SUMMARY_PROMPT = """You will receive patient data between <data> and </data> tags. Summarize the patient information provided without making any assumptions or conclusions. Include important neuropsychological battery summary scores whenever availale. Do not use bullet points, numbered lists, or section headings; craft coherent paragraphs and transition naturally between topics. Write in continuous prose using complete sentences.

# <data>
# {patient}
# </data>
# """

# PATIENT_SUMMARY_PROMPT = """You will be provided with patient data enclosed within <data> and </data> tags. Carefully summarize the information without making any assumptions or drawing conclusions. Your summary should address each subsection contained in the data, including Subject Demographics, Subject Family History, Subject Medications, Subject Health History, Physical findings, His and CVD, Unified Parkinson's Disease Rating Scale (UPDRS), Neuropsychiatric Inventory Questionnaire (NPI-Q), Geriatric Depression Scale (GDS), Functional Assessment Scale (FAS), Physical and Neurological Exam Findings, Neuropsychological Battery Summary Scores, Clinician-Assessed Medical Conditions, Genetic Testing, CSF evidence and Imaging Evidence when available. Do not mention any subsection if not available. Be sure to include any relevant neuropsychological battery summary scores if they are present. Do not use bullet points, numbered lists, or section headings; craft coherent paragraphs and transition naturally between topics. Write in continuous prose using complete sentences. Ensure that your response does not exceed 2000 tokens.

# <data>
# {patient}
# </data>
# """

# PATIENT_SUMMARY_PROMPT = """You will be provided with patient data enclosed within <data> and </data> tags. Carefully summarize the information without making any assumptions, diagnosis or drawing conclusions. Your job is only to summarize the information provided. Your summary should address each subsection contained in the data: {json_keys}. Be sure to include any relevant neuropsychological battery summary scores if they are present. Do not use bullet points, numbered lists, or section headings; instead craft coherent paragraphs and transition naturally between topics. Write in continuous prose using complete sentences. Ensure that your response does not exceed 2,000 tokens.

# <data>
# {patient}
# </data>
# """

# NACC
PATIENT_SUMMARY_PROMPT = """You will receive patient data enclosed within <data> and </data> tags. Carefully summarize this information accurately and objectively, without making any assumptions, interpretations, diagnoses, or conclusions. Focus solely on what is explicitly stated in the data. Your summary should address each subsection listed: {json_keys}. If neuropsychological battery summary scores are present, incorporate them appropriately. Present your summary in continuous prose using complete sentences. Do not use bullet points, numbered lists, or section headings; instead, craft coherent paragraphs and transition naturally between topics. Ensure that your response does not exceed 2,000 tokens.

<data>
{patient}
</data>
"""

parser = argparse.ArgumentParser(description="Process model parameters.")
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')
parser.add_argument('--max_new_tokens', type=int, default=2000, help='max_new_tokens')
parser.add_argument('--max_model_len', type=int, default=10000, help='Maximum model length')
parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Model ID')
parser.add_argument('--n_devices', type=int, default=4, help='Number of devices')
parser.add_argument('--start_id', type=int, default=0, help='Start ID')
parser.add_argument('--end_id', type=int, default=10000000, help='End ID')
parser.add_argument('--directory_path', type=str, default="/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/pretrain_summaries", help='Directory path')
parser.add_argument('--data_path', type=str, default="data.csv", help='Data path')
parser.add_argument('--output_name', type=str, default="summary.csv", help='JSON output file name')

# Parse arguments
args = parser.parse_args()
output_path = f"{args.directory_path}/{args.output_name}"

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
    """This is a function to generate LLM summaries using vllm 

    Args:
        llm: LLM object
        tokenizer: Huggingface tokenizer
        messages (List): A list of input texts / prompts
        max_new_tokens: maximum new tokens to generate

    Returns:
        List: A list of generated responses
    """
    
    prompts = []
    responses = []
    
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        max_tokens=max_new_tokens,
        presence_penalty=0.5,
        # frequency_penalty=0.5,
        # stop=stop_tokens
    )
    
    outputs = llm.chat(
        messages, 
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},  # Set to False to strictly disable thinking
    )
    
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        prompts.append(tokenizer.decode(output.prompt_token_ids))
        generated_text = output.outputs[0].text
        responses.append(generated_text)
        
    # print('Successfully finished generating', len(prompts), 'samples!')
    
    return prompts, responses

def create_batches(dataframe):
    """Yield successive n-sized batches from dataframe."""
    for i in range(0, len(dataframe), args.batch_size):
        yield dataframe.iloc[i:i + args.batch_size]


def generate_summary(patient_files, llm, tokenizer, max_new_tokens=2048):
    """
    Generate summaries for patient data using a language model.
    
    Args:
    - patient_files: List of strings containing JSON-encoded patient data.
    - llm, tokenizer: Language model and tokenizer for generating summaries.
    - max_new_tokens: maximum new tokens to generate

    Returns:
    - List of visit summaries.
    """
    messages = []
    for patient_file in patient_files:
            
        # patient_file = json.dumps(patient_file_json)
        json_keys = ", ".join(json.loads(patient_file).keys())
        question = PATIENT_SUMMARY_PROMPT.format(patient=patient_file, json_keys=json_keys)
        
        if "Qwen3" in args.model_id:
            message = [
                {"role": "user", "content": question}
            ]
        elif "Qwen" in args.model_id:
            message = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        else:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        messages.append(message)
        
    prompts, generated_summaries = get_vllm_summary(llm, tokenizer, messages, max_new_tokens=max_new_tokens)
    
    return prompts, generated_summaries
            
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
        
        prompts, generated_summaries = generate_summary(query_texts, llm, tokenizer, max_new_tokens=args.max_new_tokens)
        print('Successfully finished generating', len(generated_summaries), 'samples!')
                
        data_to_save = []
        for i, row in batch.iterrows():
            summary_data = dict(row)
            summary_data['visit_summary_prompt'] = prompts[i - batch.first_valid_index()]
            summary_data['visit_summary'] = generated_summaries[i - batch.first_valid_index()]
            data_to_save.append(summary_data)

        # Convert list of dictionaries to DataFrame
        df_to_save = pd.DataFrame(data_to_save)

        # Append DataFrame to CSV, creating the file if it does not exist yet
        df_to_save.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path), encoding='utf-8')
    
    # Destroying vllm instance
    destroy_model_parallel()
    destroy_distributed_environment()
    # del llm.llm_engine.model_executor
    del llm
    # with contextlib.suppress(AssertionError):
    #     torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
        
        