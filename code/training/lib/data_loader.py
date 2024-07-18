# coding=utf-8

import json
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
from utils.utils import load_json


def data_loader(hf_repo):
    """
    Load the pre-training dataset from Hugging Face
    :param hf_repo: the dataset in the Hugging Face
    :return dataset: the training dataset
    """
    dataset = load_dataset(hf_repo, split="train")

    return dataset

def data_loader_from_json(config, tokenizer, n=10000, split=False):
    """
    Load the fine-tuning dataset from a json file and split into training and validation sets.
    :param config: the configurations
    :param tokenizer: tokenizer to format chat template
    :param n: number of data points to load
    :return: dictionary of training and validation datasets
    """
    dataset_path = config.get("dataset_path")
    sysmsg = config.get("sysmsg")
    # usermsg_prefix = config.get("usermsg_prefix")
    user = config.get("user")
    assistant = config.get("assistant")

    data = load_json(dataset_path)
    data_df = pd.DataFrame(data[:min(n, len(data))])
    data_df = data_df.sample(frac=1, random_state=0).reset_index(drop=True)
    dataset = {}

    def format_chat_template(row):
        row_json = [
            {"role": "system", "content": sysmsg},
            # {"role": "user", "content": f"{usermsg_prefix} {row[user]}"},
            {"role": "user", "content": f"### Instruction:\n{row['instruction']}\n\n### Input:\n{row[user]}"},
            {"role": "assistant", "content": f"\n\n### Response:\n{row[assistant]}"}
        ]
        row["text"] = tokenizer.apply_chat_template(
            row_json, 
            add_generation_prompt=False,
            tokenize=False
        )
        return row

    if split:
        dataset['train'], dataset['val'] = train_test_split(data_df, test_size=0.20, random_state=42)

        dataset['train'] = Dataset.from_pandas(dataset['train'][[user, assistant]])
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=4,
        )
        
        dataset['val'] = Dataset.from_pandas(dataset['val'][[user, assistant]])
        dataset['val'] = dataset['val'].map(
            format_chat_template,
            num_proc=4,
        )

        print("Training set size:", len(dataset['train']))
        print("Validation set size:", len(dataset['val']))
    else:
        dataset['train'] = Dataset.from_pandas(data_df[[user, assistant, 'instruction']])
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=4,
        )

        print("Training set size:", len(dataset['train']))

    return dataset
