# ============================================================
#               LLM-Based Answer Extraction Script
# ============================================================
# Author: Sahana Kowshik
# Date: 2025-05-07
# ============================================================

# -------------------- Environment Setup ---------------------
import os
os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
os.environ['VLLM_SKIP_P2P_CHECK'] = "1"

# -------------------- Library Imports -----------------------
import argparse
import json
import random
import re
import string
import time
import warnings
from collections import OrderedDict
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams

# ---------------------- Model Settings -----------------------
MODEL_ID = 'Qwen/Qwen3-8B'
N_DEVICES = 2
MAX_MODEL_LEN = 10000

# ------------------ Load LLM and Tokenizer -------------------
def load_model(model_id, n_devices, max_model_len):
    """Load VLLM model and Huggingface tokenizer."""
    llm = LLM(
        model=model_id,
        tokenizer=model_id,
        tensor_parallel_size=n_devices,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        enable_lora=False,
        distributed_executor_backend='mp',
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return llm, tokenizer

# Initialize model and tokenizer
llm, tokenizer = load_model(MODEL_ID, N_DEVICES, MAX_MODEL_LEN)


# ============================================================
#                   Text Generation Utilities
# ============================================================

def get_vllm_summary(llm, tokenizer, messages, max_new_tokens=3000):
    """
    Generate LLM-based summaries using vLLM.

    Args:
        llm: LLM object
        tokenizer: Huggingface tokenizer
        messages: List of chat-formatted messages
        max_new_tokens: Max number of tokens to generate

    Returns:
        List of generated responses
    """
    prompts = [
        tokenizer.apply_chat_template(
            msg,
            add_generation_prompt=True,
            tokenize=False,
            continue_final_message=False,
            enable_thinking=False,
        )
        for msg in messages
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        max_tokens=max_new_tokens,
    )

    completions = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in completions]

    return responses


def generate_summary(answers_dicts, llm, tokenizer, max_new_tokens=2048):
    """
    Generate structured answers using LLM from a list of answer dicts.

    Args:
        answers_dicts: List of dicts with 'answer', 'question', 'option'
        llm: LLM object
        tokenizer: Tokenizer object
        max_new_tokens: Max generation tokens

    Returns:
        List of generated answer strings
    """
    messages = [
        [{"role": "user", "content": EXTRACT_ANSWER_PROMPT.format(
            answer=d['answer'], question=d['question'], option=d['option'])}]
        for d in answers_dicts
    ]
    return get_vllm_summary(llm, tokenizer, messages, max_new_tokens)


# ============================================================
#                   Answer Extraction Helpers
# ============================================================

def extract_answer_letter(text):
    """
    Extract single letter answer from 'ANSWER: X' format.
    """
    match = re.search(r'ANSWER:\s*([A-Z])', text)
    return match.group(1) if match else None


def extract_final_answer_with_answer_tag(text):
    """
    Extract answer from structured <answer>...</answer> tags.
    """
    match = re.search(r'<answer>\n(Answer: )([a-zA-Z])\n</answer>', text, re.DOTALL)
    return match.group(2).strip().upper() if match else 'invalid'


# ============================================================
#                    DataFrame Transformation
# ============================================================

def extract_naccid(row):
    row['ID'] = row['problem']['ID']
    return row


def load_results(file_path):
    """
    Load JSONL results into structured DataFrame.
    """
    results = [json.loads(line) for line in open(file_path)]
    df = pd.DataFrame(results).explode(['generated_text', 'finish_reason'])
    df = df.apply(extract_naccid, axis=1)
    df['UNQ_ID'] = range(len(df))
    return df


def extract_answers_regex(results_df, name, option_keys):
    """
    Apply regex-based extraction to predict answers.
    """
    results_df['prediction'] = results_df['generated_text'].apply(extract_final_answer_with_answer_tag)
    results_df['ground_truth'] = [p['ground_truth'] for p in results_df.problem]
    invalid_list[name] = results_df[results_df['prediction'] == 'invalid']
    invalid_before[name] = len(results_df[(results_df['prediction'] == 'invalid') |
                                          (~results_df['prediction'].isin(option_keys))])

    results_df = results_df[
        (results_df['prediction'] != 'invalid') &
        (results_df['prediction'].isin(option_keys))
    ].reset_index(drop=True)

    results_df.index = results_df.groupby('ID', sort=False).ngroup()
    return results_df


def extract_answers_regex_llm(results_df, name, option_keys):
    """
    Attempt regex first, then fallback to LLM-based answer extraction for invalid entries.
    """
    results_df['prediction'] = results_df['generated_text'].apply(extract_final_answer_with_answer_tag)
    invalid_df = results_df[
        (results_df['prediction'] == 'invalid') |
        (~results_df['prediction'].isin(option_keys))
    ].reset_index(drop=True)
    invalid_df['extraction_type'] = "llm"

    valid_df = results_df[~results_df['UNQ_ID'].isin(invalid_df['UNQ_ID'])].reset_index(drop=True)
    valid_df['extraction_type'] = "regex"

    assert len(valid_df) + len(invalid_df) == len(results_df)

    invalid_before[name] = len(invalid_df)
    options = [options_list for _ in range(len(invalid_df))]
    answers = list(invalid_df['generated_text'])
    questions = [p['question'] for p in invalid_df['problem']]

    answer_dicts = [
        {'answer': answers[i], 'option': options[i], 'question': questions[i]}
        for i in range(len(invalid_df))
    ]

    extracted_answers = generate_summary(answer_dicts, llm, tokenizer, max_new_tokens=100)
    extracted_letters = [extract_answer_letter(ans) for ans in extracted_answers]
    invalid_df['prediction'] = extracted_letters
    llm_answers[name] = extracted_answers

    results_df = pd.concat([valid_df, invalid_df], axis=0).sort_values(by='ID').reset_index(drop=True)
    results_df['ground_truth'] = [p['ground_truth'] for p in results_df.problem]
    results_df.index = results_df.groupby('ID', sort=False).ngroup()
    
    return results_df


# ============================================================
#                   Evaluation & Metrics
# ============================================================

def modify(results_df):
    """
    Prepare final evaluation DataFrame with correctness column.
    """
    results = results_df[['prediction', 'ground_truth', 'ID']].copy()
    results['correctness'] = results['prediction'] == results['ground_truth']
    results = results.reset_index(names=['problem']).reset_index(drop=True)
    return results


def pass_at_k(df, n, k=1):
    """
    Compute pass@k metric for grouped predictions.
    """
    cs = df.groupby('problem').sum('correctness')
    vals = []
    for _, row in cs.iterrows():
        c = row['correctness']
        if n - c < k:
            vals.append(1.0)
        else:
            vals.append(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
    return np.mean(vals)


def get_passat1(df):
    return df.groupby('problem').mean('correctness')[['correctness']].mean().iloc[0]


def get_consatk(df, n):
    return (
        df.groupby('problem')['prediction']
        .apply(lambda x: x.mode()[0]) ==
        df[['problem', 'ground_truth']].drop_duplicates('problem')['ground_truth'].reset_index(drop=True)
    ).sum() / n


def combine_results(results_dict, n, k):
    """
    Combine evaluation metrics into a comparison DataFrame.
    """
    final_dict = {'metric': ['pass@1', 'cons@k']}
    for key, df in results_dict.items():
        p = len(df) // n
        final_dict[key] = [pass_at_k(df, p, k), get_consatk(df, n)]

    return pd.DataFrame(final_dict)


# ============================================================
#                  Clinical Diagnostic Logic
# ============================================================

def add_clinical_diag(row):
    row['ID'] = row['NACCID']
    if (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]
    ):
        row["prediction"] = 'G'
    elif (row['NACCLBDP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]
    ):
        row["prediction"] = 'F'
    elif (row['NACCALZP'] in [1, 2]) and (
        row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]
    ):
        row["prediction"] = 'E'
    elif (row['NACCALZP'] in [1, 2]) and (row['NACCLBDP'] in [1, 2]):
        row["prediction"] = 'D'
    elif row['FTLDMOIF'] in [1, 2] or row['FTLDNOIF'] in [1, 2] or row['FTDIF'] in [1, 2]:
        row["prediction"] = 'C'
    elif row['NACCLBDP'] in [1, 2]:
        row["prediction"] = 'B'
    elif row['NACCALZP'] in [1, 2]:
        row["prediction"] = 'A'
    else:
        row["prediction"] = 'H'
    return row


def get_mean_length(df):
    """
    Compute average word length in generated outputs.
    """
    return df['generated_text'].apply(lambda x: len(x.split())).mean()


# ============================================================
#                         Prompt Template
# ============================================================

EXTRACT_ANSWER_PROMPT = """You will be given a response enclosed within <response> and </response> tags. Your task is to extract the final answer that matches one of the options listed between <options> and </options>, based on the question provided between <question> and </question>. Do not interpret or infer beyond the provided text. Return your answer strictly in this format: ANSWER: <option letter>. Do not output anything else.

<response>
{answer}
</response>

<question>
{question}
</question>

<options>
{option}
</options>"""

# ============================================================
#              Model Paths and Results Initialization
# ============================================================


test_data = pd.read_csv("/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/test_summary.csv")
test_data_summary = pd.read_csv("/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/data/nacc/training_data/testing_data_grpo/test_np.csv")

# all cases
qwen3b_results_path = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-14T120903_66b9abb12f994a55/test_np_output.jsonl"
qwen3b_drgrpo_results_path = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-14T134203_7889dfbd9d2248b8/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-14T161034_e68454cb1ec14c7c/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_cont = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-14T173914_43e267892eff4a5a/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_cont_lr_1e7 = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-16T185401_aacb12816a024788/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_wait_okay = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-16T075844_19284013be514e94/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_wait_okay_cont = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-16T111120_24e01174f88d4f09/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_wait_okay_add_okay = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-16T201218_c23bf050148d4f8b/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_wait_okay_cont_add_okay = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-16T210516_607bd8f2f9434b17/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_wait = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-18T040113_91f5cedf56924d1d/test_np_output.jsonl"
qwen3b_drgrpo_results_path_selected_wait_cont = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-18T055022_ba13620fee31458a/test_np_output.jsonl"

qwen7b_results_path = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-14T190933_450d84c246d940ee/test_np_output.jsonl"
qwen14b_results_path = "/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-16T085535_96319dfb64984083/test_np_output.jsonl"
qwen32b_results_path = '/projectnb/vkolagrp/skowshik/foundation_adrd/adrd-foundation-model/adrd_simplified_evaluation/results_2/test_np/2025-05-20T092347_8694a81cdd0a4829/test_np_output.jsonl'

# Example paths (truncated for clarity)
models_dict = {
    'qwen3b': qwen3b_results_path,
    'qwen3b_drgrpo': qwen3b_drgrpo_results_path,
    'qwen3b_drgrpo_selected': qwen3b_drgrpo_results_path_selected,
    'qwen3b_drgrpo_selected_cont': qwen3b_drgrpo_results_path_selected_cont,
    'qwen7b': qwen7b_results_path,
    'qwen14b': qwen14b_results_path,
    'qwen32b': qwen14b_results_path,
}

invalid_before = {key: 0 for key in models_dict}
invalid_after = {key: 0 for key in models_dict}
llm_answers = {key: 0 for key in models_dict}
invalid_list = {key: 0 for key in models_dict}

model_dfs = {name: load_results(path) for name, path in models_dict.items()}
option_keys = re.findall(r'\b([A-Z])\.', model_dfs['qwen3b'].iloc[0]['problem']['options'])
options_list = model_dfs['qwen3b'].iloc[0]['problem']['options']

# Extract answers
model_dfs = {
    name: extract_answers_regex_llm(df, name, option_keys)
    for name, df in model_dfs.items()
}

# Add clinician labels
test_data_subset = test_data[test_data['NACCID'].isin(model_dfs['qwen3b']['ID'])].reset_index(drop=True)
test_data_summary_subset = test_data_summary[test_data_summary['ID'].isin(model_dfs['qwen3b']['ID'])].reset_index(drop=True)
test_data_subset = test_data_subset[['NACCID', 'NACCALZP', 'NACCLBDP', 'FTLDMOIF', 'FTLDNOIF', 'FTDIF']]
test_data_subset = test_data_subset.apply(add_clinical_diag, axis=1)
test_data_subset.drop(['NACCID'], axis=1, inplace=True)
test_data_merged = test_data_summary_subset.merge(test_data_subset, on=['ID'])
model_dfs['clinician'] = test_data_merged

# Evaluate results
modified_models = {name: modify(df) for name, df in model_dfs.items()}
n = len(modified_models['clinician'])
eval_df = combine_results(modified_models, n, k=1)

# ============================================================
#                       Plotting Results
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

df_long = eval_df.melt(id_vars='metric', var_name='model', value_name='score')
sorted_models = df_long[df_long["metric"] == "pass@1"].sort_values("score")["model"].tolist()

df_sorted = pd.concat([
    pd.concat([
        df_long[(df_long["model"] == model) & (df_long["metric"] == "pass@1")],
        df_long[(df_long["model"] == model) & (df_long["metric"] == "cons@k")]
    ])
    for model in sorted_models
], ignore_index=True)

palette = {}
key_models = ['qwen3b', 'qwen7b', 'qwen14b', 'qwen32b']
yellow_shades = sns.color_palette("Reds", len(key_models))
for model, color in zip(key_models, yellow_shades):
    palette[model] = color

other_models = [m for m in df_sorted['model'].unique() if m not in key_models and m != 'clinician']
blue_shades = sns.color_palette("Blues", len(other_models))
for model, color in zip(other_models, blue_shades):
    palette[model] = color

# Plotting
plt.figure(figsize=(20, 10))
ax = sns.barplot(data=df_sorted[df_sorted['model'] != 'clinician'], x='metric', y='score', hue='model', palette=palette)

# Clinician reference
clinician_df = df_sorted[df_sorted['model'] == 'clinician']
for metric in clinician_df['metric'].unique():
    clinician_score = clinician_df[clinician_df['metric'] == metric]['score'].values[0]
    ax.axhline(y=clinician_score, linestyle='--', color='red', alpha=0.7, label='Clinician')
    break

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='edge', fontsize=9)

plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Comparison on Neuropath")
plt.legend(title='Model', loc='upper left', bbox_to_anchor=(-0.4, 1.15), frameon=False)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("np.pdf", format='pdf', bbox_inches='tight', dpi=300)
