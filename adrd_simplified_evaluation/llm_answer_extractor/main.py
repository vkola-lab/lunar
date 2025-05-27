# Entry Point
# main.py


import os
os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
os.environ['VLLM_SKIP_P2P_CHECK'] = "1"

import pandas as pd
import torch
# import gc
# import llm_wrapper
# import llm_wrapper

# from vllm.distributed.parallel_state import (
#     llm_wrapper,
#     llm_wrapper,
# )
from utils.config_loader import load_config
from models.llm_interface import LLMWrapper
from models.answer_extractor import AnswerExtractor
from utils.data_utils import load_results, prepare_test_data
from pipeline.evaluator import Evaluator
from plots.plot_results import plot_comparison



def main():
    config = load_config("config.yml")

    # Load model
    llm_wrapper = LLMWrapper(config["llm"])

    # Load model outputs
    model_dfs = {
        name: load_results(path)
        for name, path in config["models"].items()
    }

    # Extract valid answer options
    option_keys, options_list = AnswerExtractor.extract_option_keys(model_dfs)

    # Extract answers
    extractor = AnswerExtractor(llm_wrapper, option_keys, options_list)
    model_dfs = {
        name: extractor.run(df, name, config["benchmark"])
        for name, df in model_dfs.items()
    }

    # Load and prepare clinician data
    if 'neuropath' in config["benchmark"].lower():
        clinician_df = prepare_test_data(
            config["data"], model_dfs["qwen3b"]['ID'].tolist()
        )
        model_dfs['clinician'] = clinician_df

    # Evaluate
    evaluator = Evaluator(model_dfs, k=config["metrics"]["k"])
    df_scores = evaluator.evaluate()

    # Plot
    plot_comparison(df_scores, out_path=config["output"]["plot_path"], benchmark=config["benchmark"])
    
    # Delete LLM instance
    llm_wrapper.destroy_instance()
    
    # destroy_model_parallel()
    # destroy_distributed_environment()
    # del llm_wrapper.llm.llm_engine.model_executor
    # del llm_wrapper.llm
    # with contextlib.suppress(AssertionError):
    #     torch.distributed.destroy_process_group()
    # gc.collect()
    # torch.cuda.empty_cache()
    # ray.shutdown()
    # print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")


if __name__ == "__main__":
    main()