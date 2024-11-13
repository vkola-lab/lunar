# coding=utf-8

import json
import pandas as pd
import torch
import numpy as np
import pandas as pd
import re
import torchvision.models as models
from PIL import Image
from torch import nn
from datasets import load_dataset
from datasets import Dataset
from sklearn.model_selection import train_test_split
from utils.utils import load_json
from torchvision import transforms
from tokenizers import AddedToken
from utils import utils


new_tokens = set()
new_embeddings = set()
new_decoded_strings = set()

# # Load the pre-trained VGG16 model
# vgg16 = models.vgg16(pretrained=True)

# # Modify the classifier to output 4096 features directly from fc2 layer
# # We will drop the last fully connected layer and the output will be the 4096 features from fc2
# vgg16.classifier = nn.Sequential(
#     *list(vgg16.classifier.children())[:-1]  # Remove the last layer
# )


def data_loader(hf_repo):
    """
    Load the pre-training dataset from Hugging Face
    :param hf_repo: the dataset in the Hugging Face
    :return dataset: the training dataset
    """
    dataset = load_dataset(hf_repo, split="train")

    return dataset

def load_from_vgg16(path):
    # Define transformation for the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])

    # Load an image and transform
    img = Image.open(path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    # Set the model to evaluation mode
    vgg16.eval()

    # Extract features without gradient calculation
    with torch.no_grad():
        features = vgg16(batch_t)
        
    return features

def process_user_message(user_msg, reserved_tokens, tokenizer, file_type):
    # Define the regular expression pattern for words ending with .nii
    if file_type == 'mri':
        pattern = r'(/\S+\.nii)'
    elif file_type == 'emb':
        pattern = r'(/\S+\.npy)'
    else:
        raise ValueError(f"Invalid file typr {file_type}")
    
    # Define the replacement function
    # It adds "start_of image" before the word, quotes around the word, and "{reserved_tokens}end_of_mri" after
    def replace_func(match):
        matched_string = match.group(1)
        
        # Print the matched string
        # print("Matched string:", matched_string)
        return f'<|start_of_mri|>{matched_string}{reserved_tokens}<|end_of_mri|>'
    
    # Use re.sub to replace all occurrences in the input message
    processed_msg = re.sub(pattern, replace_func, user_msg)
    # print(processed_msg)
    
    return processed_msg


def add_image_tokens(df, file_type, user):
    if file_type == 'mri':
        pattern = r'(/\S+\.nii)'
    elif file_type == 'emb':
        pattern = r'(/\S+\.npy)'
    else:
        raise ValueError(f"Invalid file type {file_type}")
    all_matches = []

    # Iterate over each row in the DataFrame
    for i, row in df.iterrows():
        # Find all matches of the pattern in the 'user' column
        matches = re.findall(pattern, row[user])
        all_matches += matches
    
    all_matches = list(set(all_matches))
    if file_type == 'emb':
        for match in all_matches:
            try:
                x = np.load(match, mmap_mode='r')
            except:
                print(match)
                raise ValueError
    print(len(all_matches))
    return all_matches


def data_loader_(config, tokenizer, model, n=10000000, split=False, vision=False):
    """
    Load the fine-tuning dataset from a json file and split into training and validation sets.
    :param config: the configurations
    :param tokenizer: tokenizer to format chat template
    :param n: number of data points to load
    :return: dictionary of training and validation datasets
    """
    dataset_path = config.get("dataset_path")
    sysmsg = config.get("sysmsg")
    user = config.get("user")
    assistant = config.get("assistant")
    train_type = config.get("train_type")

    if dataset_path.endswith("json"):
        data = load_json(dataset_path)
        # data_df = pd.DataFrame(data)[:min(n, len(data))]
        data_df = pd.DataFrame.from_dict(data, orient='index')[:min(n, len(data))].reset_index()
        data_df = data_df.rename(columns={'level_0': 'ID'})
        data_df.drop("index", axis=1, inplace=True)
        data_df = data_df.sample(frac=1, random_state=0).reset_index(drop=True)  
        
    elif dataset_path.endswith("csv"):
        data_df = pd.read_csv(dataset_path).sample(frac=1, random_state=0).reset_index(drop=True)
        data_df = data_df[:min(n, len(data_df))]
        
    else:
        raise ValueError(f"Invalid input file format {dataset_path}. Please use a `json` or a `csv` file.")
        
    dataset = {}

    def format_chat_template(row):
        # Add MRI begin and end tokens in training a vision model
        if vision:
            reserved_tokens = '<|reserved_mri_token|>' * (config.get('k') - 1)
            user_msg = process_user_message(row[user], reserved_tokens, tokenizer, file_type='emb')
        else:
            user_msg = row[user]
        
        # Add llama special tokens and chat template 
        row_json = [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": utils.get_template(train_type).format(instruction=row['instruction'], input=user_msg)},
            {"role": "assistant", "content": f"\n\n### Response:\n{row[assistant]}"}
        ]
        row["text"] = tokenizer.apply_chat_template(
            row_json, 
            add_generation_prompt=False,
            tokenize=False,
            # return_tensors="pt",
        )
        return row

    if split:
        dataset['train'], dataset['val'] = train_test_split(data_df, test_size=0.20, random_state=42)

        dataset['train'] = Dataset.from_pandas(dataset['train'][[user, assistant, 'instruction']])
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=4,
        )
        
        dataset['val'] = Dataset.from_pandas(dataset['val'][[user, assistant, 'instruction']])
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
        
    # Add new tokens to tokenizer and resize embeddings only once
    if vision:
        all_matches = add_image_tokens(data_df, file_type='emb', user=user)
        if all_matches:
            for matched_string in all_matches:
                token = AddedToken(matched_string, lstrip=False, rstrip=False, normalized=False, special=False)
                new_tokens.add(token)
            tokenizer.add_tokens(list(new_tokens))
            model.resize_token_embeddings(len(tokenizer))
        
    # if vision:
    #     dataset['train'] = _prepare_non_packed_dataloader(
    #         tokenizer,
    #         model,
    #         dataset['train'],
    #         'text',
    #         max_seq_length=config.get("train_max_len"),
    #     )
    
    return dataset

def load_coco_data(config, tokenizer, model, n=100000, split=False, vision=False):
    data_df = pd.read_csv(config.get("dataset_path"))[:n]
    dataset = {}
    

    def format_chat_template(row):
        emb_name = row['image_id'].replace('train2014/', 'train2014_emb/').replace('.jpg', '.npy')
        row_json = [
            # {"role": "system", "content": sysmsg},
            {"role": "user", "content": f"### Instruction:\nProvide a caption for this image.\n\n### Input:\n<|start_of_mri|>{emb_name}<|end_of_mri|>"},
            {"role": "assistant", "content": f"\n\n### Response:\n{row['description']}"}
        ]
        row["text"] = tokenizer.apply_chat_template(
            row_json, 
            add_generation_prompt=False,
            tokenize=False,
            # return_tensors="pt",
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
        dataset['train'] = Dataset.from_pandas(data_df)
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=4,
        )

        print("Training set size:", len(dataset['train']))
        
    if vision:
        dataset['train'] = _prepare_non_packed_dataloader(
            tokenizer,
            model,
            dataset['train'],
            'text',
            max_seq_length=config.get("train_max_len"),
        )
        
    print(len(new_tokens))
        
    # Add new tokens to tokenizer and resize embeddings only once
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

        # # Update embeddings for new tokens
        # for decoded_string, embedding in zip(new_decoded_strings, new_embeddings):
        #     model.get_input_embeddings().weight.data[tokenizer.encode(decoded_string, add_special_tokens=False)[0]] = embedding
        
    def filter_empty_entries(example):
        # Adjust the field names and conditions based on your dataset schema
        return all(example[field] is not None and len(example[field]) != 0 for field in example)

    # Apply the filter function to the dataset
    dataset['train'] = dataset['train'].filter(filter_empty_entries)

    return dataset
