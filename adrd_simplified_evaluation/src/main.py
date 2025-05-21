from omegaconf import OmegaConf
import utils
import model
import torch
import gc
import contextlib
import ray

from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)

# We assume the user is logged in to HF

if __name__ == "__main__":

    # args = utils.make_parser().parse_args()

    cli_config = OmegaConf.from_cli()

    print(f"Config file: {cli_config.config_file}")

    file_config = OmegaConf.load(cli_config.config_file)

    # commandline arguments override arguments in the yaml file
    # the resulting combined config object is saved for reproducibility
    config = OmegaConf.merge(file_config, cli_config) 

    print(f"Benchmarks to run: {config.benchmarks}")


    for model_id in config.model_name:
        
        print(f"Running benchmarks for {model_id}")
        
        llm = model.load_model(config, model_id)

        for benchmark in config.benchmarks:
            
            print("Making directory for results... ", end="")
            run_path = utils.make_results_dir(config, benchmark, model_id)
            
            print(run_path)

            problems, outputs = model.run_benchmark(llm, benchmark, config, model_id)

            utils.save_results(run_path, benchmark, problems, outputs)
            
        # model.destroy_instance(llm)
        
        destroy_model_parallel()
        destroy_distributed_environment()
        del llm.llm_engine.model_executor
        del llm
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        ray.shutdown()
        print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")
