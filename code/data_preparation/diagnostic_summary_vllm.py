
#%%
import os
import torch
import argparse
import pandas as pd
import torch.distributed as dist
import json
import warnings
from vllm import LLM, SamplingParams
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
# os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
PREFIX = "./data"

#%%
def get_parser():
    parser = argparse.ArgumentParser(description="Summary Generation")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="maximum new tokens")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct", type=str, help="huggingface model name")
    parser.add_argument("--batch_size", default=2, type=int, help="Specify the batch size")
    parser.add_argument("--distributed", action="store_true", help="Set True for Distributed Training")

    return parser

#%%

def stop_token_list():
    """
    The stop token list for vLLM engine
    Note: You can add more stop tokens
    if you are using other LLMs that have stop tokens
    """
    stop_tokens = [
        "Question:",
    ]

    return stop_tokens
 
def get_summary_llama(llm, tokenizer, input_texts, device, args, system_msg):
    """This is a function to generate LLAMA summaries using vllm https://github.com/vllm-project/vllm

    Args:
        llm: LLM object
        tokenizer: Huggingface tokenizer
        input_texts (List): A list of input texts / prompts
        device (str): cuda or cpu
        system_msg (str): system message for the LLAMA prompt
        args: other arguments

    Returns:
        List: A list of generated responses
    """
    
    prompts = []
    responses = []
    
    messages = [
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": text},
        ] for text in input_texts
    ]
    
    for message in messages:
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
            # return_tensors="pt"
        )
        
        prompts.append(input_ids)
        
    stop_tokens = stop_token_list()
    
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=0.9,
        max_tokens=args.max_new_tokens,
        # stop=stop_tokens
    )
    
    completions = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    
    for i, output in enumerate(completions):
        temp_gen = output.outputs[0].text
        responses.append(temp_gen)
        
    print('Successfully finished generating', len(prompts), 'samples!')
    
    return responses

def create_batches(dataframe, batch_size):
    """Yield successive n-sized batches from dataframe."""
    for i in range(0, len(dataframe), batch_size):
        yield dataframe.iloc[i:i + batch_size]
        
        
def main(args):
    device = torch.device("cuda")
    all_nacc = pd.read_csv(f"{PREFIX}/nacc_with_summary.csv")
    json_name = f"{PREFIX}/nacc_with_llama_summaries.json"
    diag_system_msg = "For the given diagnostic information, your task is to act as a behavioral neurologist and summarize the diagnosis in 4 to 5 sentences using the 'present' tense. Focus on delivering a clear diagnostic summary based on the provided information, without mentioning the tests the subject underwent or the term 'UDS'. Ensure your summary clearly identifies the patient's cognitive status and outlines the primary etiological diagnosis, if any cognitive disorders are detected."
    patient_system_msg = "Using the information provided, please generate a comprehensive patient summary as if you are a medical professional specializing in neurology. Summarize the findings in the present tense, focusing on clear and concise language. Ensure all tests are included in the summary. Ensure that the summary is factual and based on the provided data, and does not include speculative or unnecessary information. Do not mention the term 'UDS'."
    
    
    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        dtype='bfloat16',
        tensor_parallel_size=n_devices,
        gpu_memory_utilization=0.70,
        enable_lora=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # summary_dict = {}
    
    for batch in tqdm(create_batches(all_nacc, args.batch_size)):
        torch.cuda.empty_cache()
        diag_texts = batch['diag_SUMMARY'].tolist()
        patient_texts = batch['patient_SUMMARY'].tolist()
        # print(len(diag_texts))
        
        patient_responses = get_summary_llama(llm, tokenizer, patient_texts, device, args, patient_system_msg)
        diag_responses = get_summary_llama(llm, tokenizer, diag_texts, device, args, diag_system_msg)
        for i, row in batch.iterrows():
            with open(json_name, 'a', encoding='utf-8') as json_file:
                summary_data = {
                    "index": row.name,
                    "NACCID": row["NACCID"],
                    "NACCVNUM": row["NACCVNUM"],
                    "NACCNMRI": row["NACCNMRI"],
                    "NACCMRSA": row["NACCMRSA"],
                    "NACCADC": row["NACCADC"],
                    "VISITMO": row["VISITMO"],
                    "VISITDAY": row["VISITDAY"],
                    "VISITYR": row["VISITYR"],
                    "NACCUDSD": row["NACCUDSD"],
                    "patient_SUMMARY": row["patient_SUMMARY"],
                    "diag_SUMMARY": row["diag_SUMMARY"],
                    "patient_LLAMA_SUMMARY": patient_responses[i - batch.first_valid_index()],
                    "diag_LLAMA_SUMMARY": diag_responses[i - batch.first_valid_index()]
                }
                json_file.write(json.dumps(summary_data, ensure_ascii=False) + "\n")
                json_file.close()
        torch.cuda.empty_cache()

#%%
if __name__ == "__main__":
    os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
    parser = get_parser()
    args = parser.parse_args()
    print(args.distributed)
    if args.distributed:
        n_devices = torch.cuda.device_count()
    else:
        n_devices = 1
    print(f"Using {n_devices} GPUs!")
    main(args)
