# Entry Point
# main.py


import os

os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
os.environ['VLLM_SKIP_P2P_CHECK'] = "1"

import pandas as pd
import torch
from omegaconf import OmegaConf
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
    cli_config = OmegaConf.from_cli()

    print(f"Config file: {cli_config.config_file}")
    
    file_config = OmegaConf.load(cli_config.config_file)
    main_config = OmegaConf.merge(file_config, cli_config) 
    
    # Load model
    llm_wrapper = LLMWrapper(main_config["llm"])
    
    for config_type in main_config["configs"]:
        config_path = main_config["configs"][config_type]
        config_file = OmegaConf.load(config_path)
        config = OmegaConf.merge(main_config, config_file)

        # Load model outputs
        model_dfs = {
            name: load_results(path)
            for name, path in config["models"].items()
        }

        # Extract answers
        extractor = AnswerExtractor(llm_wrapper)
        model_dfs = {
            name: extractor.run(df=df, name=name, benchmark=config["benchmark"], result_dir=config['output']['result_dir'], model_id=config["llm"]["model_id"])
            for name, df in model_dfs.items()
        }

        if 'train' in config["benchmark"].lower():
            return
        
        # Load and prepare clinician data
        if 'neuropath' in config["benchmark"].lower():
            clinician_df = prepare_test_data(
                config["data"]
            )
            model_dfs['clinician'] = clinician_df

        # Evaluate
        evaluator = Evaluator(model_dfs, k=config["metrics"]["k"])
        df_scores = evaluator.evaluate()
        
        results_csv_path = f'./{config["output"]["result_dir"]}/result_csv'
        os.makedirs(results_csv_path, exist_ok=True)
        df_scores.set_index('metric').T.sort_values(by='pass@1').reset_index().rename(columns={'index': 'model'}).to_csv(f'{results_csv_path}/{config["benchmark"]}.csv', index=False)

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