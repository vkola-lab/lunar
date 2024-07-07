# coding=utf-8

import json
import pandas as pd
from datasets import load_dataset
from datasets import Dataset
from sklearn.model_selection import train_test_split


def data_loader(hf_repo):
    """
    Load the pre-training dataset from Hugging Face
    :param hf_repo: the dataset in the Hugging Face
    :return dataset: the training dataset
    """
    dataset = load_dataset(hf_repo, split="train")

    return dataset

def data_loader_from_json(path, user, assistant, sysmsg, tokenizer, n=10000):
    """
    Load the fine-tuning dataset from a json file and split into training and validation sets.
    :param path: the path to the dataset
    :param user: column name for user messages
    :param assistant: column name for assistant messages
    :param sysmsg: system message to prepend
    :param tokenizer: tokenizer to format chat template
    :param n: number of data points to load
    :return: tuple of training and validation datasets
    """
    data = []
    with open(path, "r") as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                raise ValueError

    data_df = pd.DataFrame(data[:n])
    dataset = {}
    dataset['train'], dataset['val'] = train_test_split(data_df, test_size=0.20, random_state=42)
    
    def format_chat_template(row):
        row_json = [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": row[user].replace('\n', '').replace('**Patient Summary**', '')},
            {"role": "assistant", "content": row[assistant].replace('\n', '')}
        ]
        row["text"] = tokenizer.apply_chat_template(
            row_json, 
            add_generation_prompt=True,
            tokenize=False
        )
        return row

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

    # # Split the dataset into training and validation sets
    # train_dataset, val_dataset = train_test_split(dataset, test_size=0.20, random_state=42)

    print("Training set size:", len(dataset['train']))
    print("Validation set size:", len(dataset['val']))

    return dataset
