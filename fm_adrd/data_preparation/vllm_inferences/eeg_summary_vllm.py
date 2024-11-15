
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
    parser.add_argument("--max_new_tokens", default=512, type=int, help="maximum new tokens")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3.1-8B-Instruct", type=str, help="huggingface model name")
    parser.add_argument("--batch_size", default=4, type=int, help="Specify the batch size")
    parser.add_argument("--distributed", action="store_true", help="Set True for Distributed Training")
    parser.add_argument("--json_name", default="nacc_unique_with_llama_summaries.json", help="json file name")

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
    # print(len(messages))
    # raise ValueError
    
    for message in messages:
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
            # return_tensors="pt"
        )
        
        prompts.append(input_ids)
        
    # stop_tokens = stop_token_list()
    
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=args.max_new_tokens,
        frequency_penalty=0.5,
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
        

def generate_data(args, json_name, data, question, llm, tokenizer, device, system_msg, diag_field):
    for batch in tqdm(create_batches(data, args.batch_size)):
        patient_texts = batch['Input'].tolist()
        diag_texts = batch[diag_field].tolist()
            
        modified_diag_texts = [
            {'instruction': f"{question}", 'user': f"Answer: {diag_text}\nEEG Report:{patient_texts[i]}"} for i, diag_text in enumerate(diag_texts)
        ]
        
        diag_responses = get_summary_llama(llm, tokenizer, modified_diag_texts, device, args, system_msg=system_msg)
        
        print(f"Generating diagnosis for question: {question}")
        for i, row in batch.iterrows():
            with open(json_name, 'a', encoding='utf-8') as json_file:
                summary_data = {
                    "index": row.name,
                    "instruction": question,
                    "patient_summary": row["Input"],
                    "cognition": row["cognition"],
                    "etiology": row["etiology"],
                    "answer": diag_responses[i - batch.first_valid_index()]
                }
                json_file.write(json.dumps(summary_data, ensure_ascii=False) + "\n")
                json_file.close()
            # raise ValueError
        
        torch.cuda.empty_cache()
        
        
def main(args):
    print(args.model_id)
    PREFIX = "./data/1022"
    print(args.json_name)
    device = torch.device("cuda")
    data = pd.read_csv(f"{PREFIX}/eeg/cleaned_eeg_data.csv")
    
    print(f"Number of cases to process: {len(data)}")
    
    json_name = f"{PREFIX}/eeg/{args.json_name}"

    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=n_devices,
        gpu_memory_utilization=0.90,
        # max_model_len=30000,
        enable_lora=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(json_name)
    
    cog_data = data[~data['cognition'].isna()].reset_index()
    et_data = data[~data['etiology'].isna()].reset_index()
    
    system_msg = "You will receive an EEG report along with a corresponding answer. Your task is to respond to a specified question by looking at the Answer line and supporting your choice with details from the EEG report. When crafting your response, adopt a tone that suggests you are personally answering the question. Refrain from offering any recommendations or including serial numbers in your response."
    
    generate_data(args=args, json_name=json_name, data=cog_data, question="Assess the patient's Cognitive status at visit using the provided information. Provide the cognitive status and your rationale behind it.", llm=llm, tokenizer=tokenizer, device=device, system_msg=system_msg, diag_field='cognition')
    generate_data(args=args, json_name=json_name, data=et_data, question="Using the provided information, determine the Primary etiologic diagnosis for this patient. Provide the etiology and your rationale behind it.", llm=llm, tokenizer=tokenizer, device=device, system_msg=system_msg, diag_field='etiology')
    
    

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
