
#%%
import os
import torch
import argparse
import pandas as pd
import torch.distributed as dist
import json
import warnings
warnings.filterwarnings("ignore")
import random
import re

from tqdm import tqdm
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
# os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'


#%%
def get_parser():
    parser = argparse.ArgumentParser(description="Summary Generation")
    parser.add_argument("--max_new_tokens", default=1024, type=int, help="maximum new tokens")
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
 
def get_summary_llama(llm, tokenizer, input_texts, args, system_msg):
    """This is a function to generate LLAMA summaries using vllm https://github.com/vllm-project/vllm

    Args:
        llm: LLM object
        tokenizer: Huggingface tokenizer
        input_texts (List): A list of input texts / prompts
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
        
def modify_diag_text(oth, diag_text):
    modified_text = diag_text
    for desc in oth:
        modified_text = re.sub(r'\n\t"{desc}":\s*".*?",'.format(desc=desc.replace('(', '\(').replace(')', '\)')), '', modified_text)
        
    return modified_text

def generate_data(args, json_name, data, questions, llm, tokenizer, system_msg, oth):
    for batch in tqdm(create_batches(data, args.batch_size)):
        patient_texts = batch['patient_summary'].tolist()
        diag_texts = batch['diag_summary'].tolist()
        naccudsd = batch['NACCUDSD'].tolist()
        
        modified_diag_texts = []
        instructions = []
        for i, question in enumerate(questions):
            
            modified_diag_texts = [
                {'instruction': f"{question}", 'user': f"Diagnosis:\n{modify_diag_text(oth, diag_texts[i])}\nPatient:\n{patient_texts[i]}"} for i, diag_text in enumerate(diag_texts)
            ]
            
            diag_responses = get_summary_llama(llm, tokenizer, modified_diag_texts, args, system_msg=system_msg)
            
            print(f"Generating diagnosis for question: {question}")
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
                        "CDRGLOB": row["CDRGLOB"],
                        "instruction": question,
                        "patient_summary": row["patient_summary"],
                        "diag_summary": row["diag_summary"],
                        "answer": diag_responses[i - batch.first_valid_index()]
                    }
                    json_file.write(json.dumps(summary_data, ensure_ascii=False) + "\n")
                    json_file.close()
                # raise ValueError
        
        # torch.cuda.empty_cache()
    
def load_model(args):
    print(args.model_id)
    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=n_devices,
        gpu_memory_utilization=0.90,
        # max_model_len=30000,
        enable_lora=False,
        distributed_executor_backend='ray'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return llm, tokenizer

def load_data(args, data_path):
    data = pd.read_csv(data_path)
    end_id = min(len(data), args.end_id)
    print(args.start_id, end_id)
    data = data[args.start_id:end_id].reset_index()
    print(f"Number of cases to process: {len(data)}")
    return data

def get_diagnosis_dict(datadict_path):
    with open(datadict_path, 'r', encoding='utf-8') as f:
        datadict = json.load(f)
        
    cog_var = []
    oth = []
    for variable, info in datadict.items():
        if info['FORM ID'] == 'D1':
            desc = info['Description']
            if ('normal cognition' in desc.lower() or 'mci' in desc.lower() or 'cognitive status' in desc.lower() or 'met criteria for' in desc.lower() or 'during uds follow-up' in desc.lower()) and ('Incident' not in desc):
                # print(desc)
                cog_var.append(desc)
            else:
                oth.append(desc)
                
    diagnosis_dict = {'Cognitive Status': datadict['NACCUDSD']['Values']}
    diagnosis_dict['Cognitive Status'].pop('2')
    
    return diagnosis_dict, oth, cog_var
        
def main(args):
    data_path = "./data/1022/csv_to_txt/all_nacc_csv_to_txt.csv"
    datadict_path = "/projectnb/vkolagrp/datasets/NACC/data_dictionaries/NACC_dictionary.json"
    
    ## Load Model
    llm, tokenizer = load_model(args)
        
    ## Prepare data
    PREFIX = "./data/1105"
    print(args.json_name)
    data = load_data(args, data_path)
    if not os.path.exists(f"{PREFIX}/diagnosis"):
        os.makedirs(f"{PREFIX}/diagnosis", exist_ok=True)
    json_name = f"{PREFIX}/diagnosis/{args.json_name}"
    
    diagnosis_dict, oth, cog_var = get_diagnosis_dict(datadict_path)
    
    questions = [
        """Provide the Cognitive status at UDS visit for a patient presenting with the following information. Please Report them in JSON format according to the following dictionary.\n"""+ json.dumps(diagnosis_dict, indent=4), 
    ]
    
    system_msg = """You are a helpful behavioral neurologist AI assisstant. You will be given patient information and the response template in JSON format. Your purpose is to answer the given question, and report them in JSON format according to the provided dictionary.
Follow these guidelines when answering the provided question:
1. Please use the 'Diagnosis' section of the input to get your answers. But, do not reference any details from the 'Diagnosis' section in your responses. 
2. Please verify your answers with the 'Diagnosis' section but don't mention it in the answer.
3. Print your response following the template
{
    "Answer" (Replace this word according the asked question): {
        "value": ...,
        "explanation": ...
    }
}
Do not print anything else, only the JSON. Any explanation must be in the 'explanation' key. Base your 'explanation' only on the 'Patient' section and the 'value' from the 'Diagnosis' section. Provide a complete and elaborate explanation.
4. Use phrases such as 'I diagnose', 'I assess', and so on indicating you are the one making the assessment.
5. Include information from all sections of 'Patient' summary in your explanation.
6. Do not use the term 'Diagnosis section' in your answer.
"""
    
    generate_data(args=args, json_name=json_name, data=data, questions=questions, llm=llm, tokenizer=tokenizer, system_msg=system_msg, oth=oth)
    
    

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
