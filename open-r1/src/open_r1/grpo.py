# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import wandb

import datasets
import pandas as pd
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1.utils import utils
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


def data_loader_grpo(dataset_path, system_prompt, n=10000000, split=False, vision=False):
    """
    Load the fine-tuning dataset from a json file and split into training and validation sets.
    :param config: the configurations
    :param tokenizer: tokenizer to format chat template
    :param n: number of data points to load
    :return: dictionary of training and validation datasets
    """

    if dataset_path.endswith("json"):
        data = utils.load_json(dataset_path)
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
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utils.get_template(train_type="grpo").format(patient=row["visit_summary"], question=row['question'], options=row['options'])}
        ]
        # text = tokenizer.apply_chat_template(
        #     row_json,
        #     tokenize=False,
        #     add_generation_prompt=True,
        #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        # )
        
        # row["prompt"] = text
        row["prompt"] = row_json
        return row

    if split:
        dataset['train'], dataset['val'] = train_test_split(data_df, test_size=0.20, random_state=42)

        dataset['train'] = Dataset.from_pandas(dataset['train'])
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=128,
        )
        
        dataset['val'] = Dataset.from_pandas(dataset['val'])
        dataset['val'] = dataset['val'].map(
            format_chat_template,
            num_proc=128,
        )

        print("Training set size:", len(dataset['train']))
        print("Validation set size:", len(dataset['val']))
    else:
        dataset['train'] = Dataset.from_pandas(data_df)
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=16,
        )

        print("Training set size:", len(dataset['train']))
    
    return dataset


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    dataset = data_loader_grpo(script_args.dataset_name, system_prompt=training_args.system_prompt)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    # def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
    #     prompt = []

    #     if training_args.system_prompt is not None:
    #         prompt.append({"role": "system", "content": training_args.system_prompt})

    #     if prompt_column not in example:
    #         raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

    #     prompt.append({"role": "user", "content": example[prompt_column]})
    #     return {"prompt": prompt}

    # dataset = dataset.map(make_conversation)
    # print(dataset)
    # raise ValueError

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # print("hello")
    # print(script_args)
    # print(training_args)
    # print(model_args)
    wandb.init(
        # set the wandb project where this run will be logged
        project="open_r1",
        
        # track hyperparameters and run metadata
        # config=training_args
    )
    # wandb.init(mode="disabled")
    main(script_args, training_args, model_args)
