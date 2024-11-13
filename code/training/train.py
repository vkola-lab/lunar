#!/usr/bin/env python
# coding=utf-8
#
# MIT License
#
# Copyright (c) 2024 Kolachalama Lab at Boston University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
import sys
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import yaml
import torch
import json
import pandas as pd
import wandb
from datetime import datetime, timedelta
from huggingface_hub import login

from lib.model_loader import load_model, load_model_quant, load_model_quant_gptq
from lib.trainer_loader import load_trainer
# from lib.model_loader_eval import load_model_with_llama_proj
from lib.data_loader import data_loader_, load_coco_data
from utils.utils import CustomStream, load_config


def get_parser():
    parser = argparse.ArgumentParser(description="Finetuning")
    parser.add_argument("--mode", default=1, type=int, help="Set to 1: Train adapter, 2: Train image projection layer")
    parser.add_argument("--wandb", action="store_true", help="Set True to enable wandb")
    parser.add_argument("--n", default=10000000, type=int, required=False, help="Specify the dataset size")
    parser.add_argument("--vision", action="store_true", help="Set True to load VisionModelForCausalLM")
    parser.add_argument("--quant", action="store_true", help="Set True to load quantized model")

    return parser

def main(config, args):
    """Run the program"""
    if args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="FM_ADRD",
            
            # track hyperparameters and run metadata
            config=config
        )
        wandb.run.log_code(".")
    else:
        wandb.init(mode="disabled")

    # Retrieve the pathes of needed hyperparameters
    epochs = config.get("epochs")

    # Load the model and tokenizer
    print("Start the Fine-tuning process......")
    if args.quant:
        print("Loading quantized model")
        model, tokenizer = load_model_quant(config, vision=args.vision)
    else:
        print("Loading unquantized model")
        model, tokenizer = load_model(config, vision=args.vision)
    
    # Load dataset
    dataset = data_loader_(config, tokenizer, model, n=args.n, vision=args.vision)
    
    # print("Saving initial model weights")
    # torch.save(model.get_input_embeddings().state_dict(), f'{config.get("save_dir")}/model_input_embeddings.pth')
    # torch.save(model.get_output_embeddings().state_dict(), f'{config.get("save_dir")}/model_output_embeddings.pth')
    
    # load_model_with_llama_proj(config, tokenizer, lora_path=None, save_dir=config.get("save_dir"), torch_dtype=torch.bfloat16, push_to_hub=True)
    
    # Load Trainer
    trainer = load_trainer(
        config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_train_epochs=epochs
    )

    # Start the training process
    trainer.train()
    
    # # save_merged_model
    trainer.push_to_hub(config.get("hub_model_id"))
    
    # if args.vision:
    #     # model.model.save_pretrained(config.get("save_dir"))
    #     torch.save(model.get_input_embeddings().state_dict(), f'{config.get("save_dir")}/model_input_embeddings.pth')
    #     torch.save(model.get_output_embeddings().state_dict(), f'{config.get("save_dir")}/model_output_embeddings.pth')
    #     print(f"Saved input and output embedding layers to {config.get('save_dir')}")
        
    #     if hasattr(model.model, 'llama_proj'):
    #         print(model.model.llama_proj)
    #         torch.save(model.model.llama_proj.state_dict(), f'{config.get("save_dir")}/llama_proj.pth')
    #         print(f"Saved project layer to {config.get('save_dir')}/llama_proj.pth")
            
    #     load_model_with_llama_proj(config, tokenizer, lora_path=config.get("hub_model_id"), save_dir=config.get("save_dir"), torch_dtype=torch.bfloat16, push_to_hub=True)


if __name__ == "__main__":
    # Load the parser
    parser = get_parser()
    args = parser.parse_args()
    print(args.n)
    
    # Load the configuration
    if args.mode == 1:
        print("Training adaptor")
        if args.quant:
            print("Loading quant config")
            config = load_config(file_name="./code/training/config/config_quant.yml")
        else:
            print("Loading large config")
            config = load_config(file_name="./code/training/config/config_large.yml")
    elif args.mode == 2:
        print("Training projection layer")
        config = load_config(file_name="./code/training/config/config_imaging.yml")
    else:
        raise ValueError(f"Unknown mode argument {args.mode}. Please set mode to 1 or 2.")
    
    result_dir = config.get("result_dir")
    hf_read_token = config.get("hf_read_token")

    # get the current working directory
    cwd = os.getcwd()
    login(token=hf_read_token)  # Hugging Face Login

    # print output to the console
    print('\n\nThe current working directory is', cwd, '\n\n')

    # Check out the system assigned GPU id
    count = torch.cuda.device_count()
    print('There are', count, 'GPU/GPUs available!',
          'The devices are:', os.getenv("CUDA_VISIBLE_DEVICES"), '\n')

    # Get the current date and time
    time = datetime.now()

    # Create a subdirectory with the current date
    dir = os.path.join(result_dir, time.strftime("%Y-%m-%d"))
    os.makedirs(dir, exist_ok=True)

    # Create a log file with the exact time as the file name
    name = time.strftime("%H-%M-%S.log.txt")
    path = os.path.join(dir, name)

    # Configure the logging module to write to the log file
    logging.basicConfig(
        filename=path,
        level=logging.INFO,  # Adjust the log level as needed
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Redirect sys.stdout to the custom stream
    stream = CustomStream(path, sys.stdout)

    sys.stdout = stream
    print(yaml.dump(config, default_flow_style=False), '\n\n')
    main(config=config, args=args)
    sys.stdout = sys.__stdout__
