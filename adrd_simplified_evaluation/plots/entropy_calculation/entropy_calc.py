
import os
os.environ['VLLM_SKIP_P2P_CHECK'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
import re
import json

from tqdm import tqdm
from datetime import timedelta
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams
# from prompts import zero_shot_prompt, explanation_prompt

max_new_tokens = 5000
n_devices = 1
n_cases = 1000
enable_lora = False

# model_id = '/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample'
# save_name = "entropies/oversample.json"

# model_id = '/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_sce_tanh'
# save_name = "entropies/oversample_sce_tanh.json"

# model_id = '/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_dedup'
# save_name = "entropies/oversample_dedup.json"

# model_id = '/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/open-r1/ckpt/ckpt_access/qwen25_3B_drgrpo_gp16_nacc_inc_oversample_dedup_sce_tanh'
# save_name = "entropies/oversample_dedup_sce_tanh.json"

# model_id = 'Qwen/Qwen2.5-3B-Instruct'
# save_name = "entropies/q3b.json"

model_id = 'Qwen/Qwen2.5-7B-Instruct'
save_name = "entropies/q7b.json"


GRPO_TEMPLATE = """Question: {question}.

Answer Choices: 
{options}
"""
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


data_paths = {
    "nacc": {
        "test_cog": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/with_summary/test_cog.csv",
        "test_etpr": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/with_summary/test_etpr.csv",
        "test_pet": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/with_summary/test_pet.csv",
        "test_csf": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/with_summary/test_csf.csv",
        "test_dat": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/with_summary/test_dat.csv",
        "test_np_one": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/with_summary/test_np_one.csv",
    },
    "adni":{
        "test_cog": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/adni/training_data/testing_data_grpo/with_summary/test_cog.csv",
        "test_etpr": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/adni/training_data/testing_data_grpo/with_summary/test_etpr.csv",
        "test_pet": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/adni/training_data/testing_data_grpo/with_summary/test_pet.csv",
        "test_csf": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/adni/training_data/testing_data_grpo/with_summary/test_csf.csv",
    },
    "nifd":{
        "test_cog": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nifd/training_data/testing_data_grpo/with_summary/test_cog.csv",
        "test_etpr": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nifd/training_data/testing_data_grpo/with_summary/test_etpr.csv",
    },
    "ppmi":{
        "test_cog": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/ppmi/training_data/testing_data_grpo/with_summary/test_cog.csv",
        "test_etpr": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/ppmi/training_data/testing_data_grpo/with_summary/test_etpr.csv",
        "test_dat": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/ppmi/training_data/testing_data_grpo/with_summary/test_dat.csv",
    },
    "brainlat":{
        "test_cog": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/brainlat/training_data/testing_data_grpo/with_summary/test_cog.csv",
        "test_etpr": "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/brainlat/training_data/testing_data_grpo/with_summary/test_etpr.csv",
    }
}
data_subset_path = "random_test_data.csv"

# Load model
def load_model(model_id):
    """Load VLLM model and Huggingface tokenizer."""
    print(model_id)
    llm = LLM(
        model=model_id,
        tokenizer=model_id,
        tensor_parallel_size=n_devices,
        gpu_memory_utilization=0.4,
        max_model_len=15000,
        enable_lora=enable_lora,
        distributed_executor_backend='mp',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return llm, tokenizer

# Get generations
def get_vllm_summary(llm, tokenizer, messages, max_new_tokens):
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
            # return_tensors="pt"
        )
        
        prompts.append(input_ids)
    
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L38-L66
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        min_p=0.0,
        max_tokens=max_new_tokens,
        logprobs=20
        # frequency_penalty=0.5,
        # stop=stop_tokens
    )
    
    completions = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    return completions

# Get entropy


def compute_entropy_from_logprobs(logprobs_list):
    """
    logprobs_list: list of dicts {token_id: Logprob} per generated token
    Returns: list of per-token entropies
    """
    entropies = []
    for token_logprobs in logprobs_list:
        # Extract log probs (these are top-k, not full distribution)
        log_probs = np.array([lp.logprob for lp in token_logprobs.values()])
        probs = np.exp(log_probs)
        
        # Normalize (since we only have top-k, not full vocab)
        probs = probs / probs.sum()
        
        # Shannon entropy: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        entropies.append(entropy)
    
    return entropies


def main():
    # Load model
    print("Loading model")
    llm, tokenizer = load_model(model_id)
    
    # Loading data
    print("Loading data")
    if os.path.exists(data_subset_path):
        print("File exists")
        test_data = pd.read_csv(data_subset_path)
    else:
        from tqdm import tqdm
        loaded_dfs = {}

        for dataset_name, file_dict in tqdm(data_paths.items()):
            dfs = []
            for split_name, file_path in file_dict.items():
                dfs.append(pd.read_csv(file_path))
            # Concatenate all DataFrames for this dataset_name
            loaded_dfs[dataset_name] = pd.concat(dfs, ignore_index=True)

        # Now, loaded_dfs contains a DataFrame for each dataset_name with all splits concatenated

        test_data_parts = []
        for key, df in loaded_dfs.items():
            sample_n = min(n_cases, len(df))
            test_data_parts.append(df.sample(n=sample_n, random_state=42))  # random_state for reproducibility

        test_data = pd.concat(test_data_parts, ignore_index=True).reset_index(drop=True)
        test_data.to_csv("random_test_data.csv", index=False)

    # Creating prompts
    print("Creating prompts")
    prompts = [
        GRPO_TEMPLATE.format(question=f'{row["visit_summary"]}\n\n{row["question"]}', options=row['options'])
        for _, row in test_data.iterrows()
    ]

    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ] for prompt in prompts
    ]

    print("enerating responses")
    outputs = get_vllm_summary(llm, tokenizer, messages, max_new_tokens)

    print("Getting entropies")
    entropies = {
        "entropy": [],
        "mean": [],
        "min": [],
        "max": [],
    }
    for output, prompt in zip(outputs, prompts):
        gen = output.outputs[0]
        
        if gen.logprobs:
            entropy = compute_entropy_from_logprobs(gen.logprobs)
            # print(f"Prompt: {prompt!r}")
            entropies["entropy"].append(entropy)
            entropies["mean"].append(np.mean(entropy))
            entropies["min"].append(np.min(entropy))
            entropies["max"].append(np.max(entropy))
            
    # Save
    print("Saving the file")
    with open(save_name, "w") as f:
        json.dump(entropies, f, indent=4)


    print(f"Mean: {np.mean(entropies['mean'])}")
    print(f"Min: {np.mean(entropies['min'])}")
    print(f"Max: {np.mean(entropies['max'])}")


if __name__ == "__main__":
    main()


