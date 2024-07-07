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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime, timedelta
from huggingface_hub import login

from lib.model_loader import load_model, trainer_loader
from lib.data_loader import data_loader_from_json
from utils.utils import CustomStream, load_config


def get_parser():
    parser = argparse.ArgumentParser(description="Finetuning")
    parser.add_argument("--distributed", action="store_true", help="Set True for Distributed Training")
    parser.add_argument("--n", default=10000, type=int, required=False, help="Specify the dataset size")

    return parser


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    args.dist_url = "env://"
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ.get("RANK"))
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        
    else:
        print('Does not support training without GPU.')
        sys.exit(1)
    
    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=2700),
    )

    torch.cuda.set_device(args.gpu) 
    args.rank = args.gpu
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def cleanup():
    dist.destroy_process_group()

def main(config, args):
    """Run the program"""
    # print(args.rank, args.world_size)
    device_map = torch.device(f"cuda:{args.rank}" if args.distributed else f"cuda")
    # Retrieve the pathes of needed hyperparameters
    epochs = config.get("epochs")
    dataset_path = config.get("dataset_path")
    user = config.get("user")
    assistant = config.get("assistant")
    sysmsg = config.get("sysmsg")

    # Load the model and tokenizer
    print("Start the Fine-training process......")
    model, tokenizer = load_model(config)
    
    # Load dataset
    dataset = data_loader_from_json(dataset_path, user, assistant, sysmsg, tokenizer, n=args.n)
    
    # Load Trainer
    if args.distributed:
        model.to(device_map)
        model = DDP(model, device_ids=[args.gpu])
        
        trainer = trainer_loader(
            config,
            model=model.module,
            tokenizer=tokenizer,
            dataset=dataset,
            num_train_epochs=epochs
        )
    else:
        trainer = trainer_loader(
            config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            num_train_epochs=epochs
        )

    # Start the training process
    trainer.train()


if __name__ == "__main__":
    # Load the parser
    parser = get_parser()
    args = parser.parse_args()
    print(args.n)
    
    # Load the configuration
    config = load_config(file_name="./code/training/config/config.yml")
    result_dir = config.get("result_dir")
    hf_read_token = config.get("hf_read_token")
    
    if args.distributed:
        init_distributed_mode(args)

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
