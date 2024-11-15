
#%%
import os
import torch
import argparse
import pandas as pd
import torch.distributed as dist
import json
import warnings
import random
from vllm import LLM, SamplingParams
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
# os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'


#%%
def get_parser():
    parser = argparse.ArgumentParser(description="Summary Generation")
    parser.add_argument("--max_new_tokens", default=16000, type=int, help="maximum new tokens")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help="huggingface model name")
    parser.add_argument("--batch_size", default=4, type=int, help="Specify the batch size")
    parser.add_argument("--distributed", action="store_true", help="Set True for Distributed Training")
    parser.add_argument("--json_name", default="nacc_unique_with_llama_summaries.json", help="json file name")
    parser.add_argument("--start_id", default=0, type=int, help="start id")
    parser.add_argument("--end_id", default=100000, type=int, help="end id")

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
            {"role": "user", "content": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{text['instruction']}\n\n### Input:\n{text['user']}"},
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
        temperature=0.1,
        max_tokens=args.max_new_tokens,
        # stop=stop_tokens
    )
    
    completions = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    
    for i, output in enumerate(completions):
        temp_gen = output.outputs[0].text.replace(" (NINDS/AIREN criteria)", "").replace(" UDS", "")
        responses.append(temp_gen)
        
    print('Successfully finished generating', len(prompts), 'samples!')
    
    return responses

def create_batches(dataframe, batch_size):
    """Yield successive n-sized batches from dataframe."""
    for i in range(0, len(dataframe), batch_size):
        yield dataframe.iloc[i:i + batch_size]
        
def main(args):
    PREFIX = "./data/1013"
    
    print(f"Loading model {args.model_id}...")
    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        dtype='bfloat16',
        tensor_parallel_size=n_devices,
        gpu_memory_utilization=0.90,
        enable_lora=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(args.json_name)
    device = torch.device("cuda")
    all_nacc = pd.read_csv(f"{PREFIX}/csv_to_txt/all_nacc_csv_to_txt.csv")
    end_id = min(len(all_nacc), args.end_id)
    print(args.start_id, end_id)
    all_nacc = all_nacc[args.start_id:end_id]
    print(f"Number of cases to process: {len(all_nacc)}")
    json_name = f"{PREFIX}/llama_summaries/patient/{args.json_name}"
    
    print(json_name)
    print("Generating patient prompts")
    
    for batch in tqdm(create_batches(all_nacc, args.batch_size)):
        patient_texts = batch['patient_SUMMARY'].tolist()
        
        modified_patient_texts = [
            {'instruction': "Using the information provided, please generate a complete and comprehensive medical history report of the patient presenting with the following information. Do not provide any 'clinical diagnosis'. Include ranges in your summary. Do not include patient's name. Do not mention the term 'UDS'. Exclude the information that say not applicable or not available or unknown. Ensure the summary includes Subject Demographics, Subject Family History, Subject Medications, Subject Health History, Physical, HIS and CVD, UPDRS, Neuropsychiatric Inventory Questionnaire, Geriatric Depression Scale, Functional Assessment Scale, Physical/ Neurological Exam Findings, Neuropsychological battery Summary Scores, Mini-mental state examination score (MMSE), Montreal Cognitive Assessment score (MoCA), Genetic testing, MRI imaging evidence and PET imaging evidence. Include ranges in your summary. Include the test numbers whenever available. Do not include patient's name. Do not mention the term 'UDS'. Do not provide any 'clinical diagnosis'.", 'user': summary} for summary in patient_texts
        ]
        
        patient_responses = get_summary_llama(llm, tokenizer, modified_patient_texts, device, args, system_msg="You are a helpful behavioral neurologist AI assistant.")
        
        for i, row in batch.iterrows():
            with open(json_name, 'a', encoding='utf-8') as json_file:
                summary_data = {
                    "index": row.name,
                    "NACCID": row["NACCID"],
                    "NACCVNUM": row["NACCVNUM"],
                    "NACCMNUM": row["NACCMNUM"],
                    "NACCNMRI": row["NACCNMRI"],
                    "NACCMRSA": row["NACCMRSA"],
                    "NACCADC": row["NACCADC"],
                    "VISITMO": row["VISITMO"],
                    "VISITDAY": row["VISITDAY"],
                    "VISITYR": row["VISITYR"],
                    "MRIMO": row["MRIMO"],
                    "MRIDY": row["MRIDY"],
                    "MRIYR": row["MRIYR"],
                    "NACCUDSD": row["NACCUDSD"],
                    "NACCETPR": row["NACCETPR"],
                    "patient_SUMMARY": row["patient_SUMMARY"],
                    "diag_SUMMARY": row["diag_SUMMARY"],
                    "patient_LLAMA_SUMMARY": patient_responses[i - batch.first_valid_index()]
                }
                json_file.write(json.dumps(summary_data, ensure_ascii=False) + "\n")
                json_file.close()
            # raise ValueError
        
        torch.cuda.empty_cache()

#%%
if __name__ == "__main__":
    os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/hub/'
    parser = get_parser()
    args = parser.parse_args()
    print(args.distributed)
    if args.distributed:
        n_devices = torch.cuda.device_count()
    else:
        n_devices = 1
    print(f"Using {n_devices} GPUs!")
    main(args)
