
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
        

def generate_data(args, json_name, data, questions, llm, tokenizer, system_msg):
    for batch in tqdm(create_batches(data, args.batch_size)):
        patient_texts = batch['patient_summary'].tolist()
        diag_texts = batch['diag_summary'].tolist()
        naccudsd = batch['NACCUDSD'].tolist()
        
        modified_diag_texts = []
        instructions = []
        for i, question in enumerate(questions):
            
            modified_diag_texts = [
                {'instruction': f"{question}", 'user': f"Patient:{patient_texts[i]}\nDiagnosis (DO NOT MENTION ANYTHING FROM THIS SECTION IN YOUR ANSWERS. IT IS ONLY FOR YOUR REFERENCE): {diag_text}"} for i, diag_text in enumerate(diag_texts)
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
        
        torch.cuda.empty_cache()


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
        
def main(args):
    datadict_path = "/projectnb/vkolagrp/datasets/NACC/data_dictionaries/NACC_dictionary.json"
    
    ## Load Model
    llm, tokenizer = load_model(args)
        
    ## Prepare data
    PREFIX = "./data/1022"
    data_path = f"{PREFIX}/csv_to_txt/all_nacc_csv_to_txt.csv"
    print(args.json_name)
    data = load_data(args, data_path)
    if not os.path.exists(f"{PREFIX}/diagnosis"):
        os.makedirs(f"{PREFIX}/diagnosis", exist_ok=True)
    json_name = f"{PREFIX}/diagnosis/{args.json_name}"
    
    nc_questions = [
        "Given the following patient information, compile a brief and efficient hypothesis-driven neurological history in a paragraph.", 
        "Assess the patient's Cognitive status at UDS visit using the provided information, categorizing it as normal cognition, mild cognitive impairment, or dementia.", 
    ]
    imp_questions = [
        "Given the following patient information, compile a brief and efficient hypothesis-driven neurological history in a paragraph.", 
        "Assess the patient's Cognitive status at UDS visit using the provided information, categorizing it as normal cognition, mild cognitive impairment, or dementia.", 
        "Using the provided information, determine the Primary etiologic diagnosis for this patient.", 
        "Evaluate whether psychiatric disorders are contributing to this patient's neurological condition and explain your reasoning."
    ]
    de_questions = [
        "Given the following patient information, compile a brief and efficient hypothesis-driven neurological history in a paragraph.", 
        "Assess the patient's Cognitive status at UDS visit using the provided information, categorizing it as normal cognition, mild cognitive impairment, or dementia.", 
        "Using the provided information, determine the Primary etiologic diagnosis for this patient.", 
        "Determine the differential diagnosis for dementia that best aligns with the patient's symptoms and presentation. List the three most probable diagnoses without specifying 'top 3' in your answer.", 
        "Evaluate whether psychiatric disorders are contributing to this patient's neurological condition and explain your reasoning."
    ]
    
    nc_data = data[data['NACCUDSD'] == 1].reset_index()
    imp_data = data[(data['NACCUDSD'] == 2) | (data['NACCUDSD'] == 3)].reset_index()
    de_data = data[data['NACCUDSD'] == 4].reset_index()
    
    # system_msg = "You are a behavioral neurologist AI assistant.  If any information is not available, assume that information was not collected. Do not include any recommendations in your answer. Follow a patient-centric approach to answer the given question. Answer responsibly, avoiding overconfidence, and encourage the user to consult a healthcare professional for advice. Do not use the phrases like 'clinician's assessment' or 'clinical diagnosis' or 'a clinician diagnosed' or 'a consensus panel diagnosed' in your answers. Answer the question as though you are providing the answers when presented with only patient information."
    system_msg = "You are a behavioral neurologist. Follow these conditions when generating the answer:\n1. Your task is to answer the question by basing your response solely on the 'Patient' section. 'Diagnosis' section is only provided for your reference. It is not patient history. Use it only to verify your analysis. Always verify your answers with the 'Diagnosis' section but don't mention it in the answer. \n2. You are allowed to use the 'Diagnosis' section of the input indirectly to guide your answers. But, do not reference any details from the 'Diagnosis' section in your responses.\n3. Do not use terms like 'clinician', 'clinician's assessment', 'clinical diagnosis', 'deemed by a clinician' or any other phrases indicating a confirmed diagnosis. Instead use phrases such as 'I diagnose', 'I assess', and so on indicating you are the one making the assessment.\n4. Answer responsibly, avoiding overconfidence, and encourage the user to consult a healthcare professional for advice.\n5. Your responses should not include any recommendations.\n6. When any information is missing, assume it was not collected.\n7. Please verify your answers with the 'Diagnosis' section in the input."
    
    generate_data(args=args, json_name=json_name, data=nc_data, questions=nc_questions, llm=llm, tokenizer=tokenizer, system_msg=system_msg)
    generate_data(args=args, json_name=json_name, data=imp_data, questions=imp_questions, llm=llm, tokenizer=tokenizer, system_msg=system_msg)
    generate_data(args=args, json_name=json_name, data=de_data, questions=de_questions, llm=llm, tokenizer=tokenizer, system_msg=system_msg)
    
    

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
