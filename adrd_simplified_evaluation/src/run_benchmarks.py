from omegaconf import OmegaConf
import utils
import llm_interface

if __name__ == "__main__":

    # Commandline arguments override arguments in the yaml file
    # the resulting combined config object is saved for reproducibility
    cli_config = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_config.config_file)
    config = OmegaConf.merge(file_config, cli_config) 

    print(f"Config file: {cli_config.config_file}")
    print(f"Model: {config.LLM.model}")
        
    llm = llm_interface.LLMWrapper(config)

    for benchmark_path in config.benchmarks.benchmark_list:

        run_path = utils.make_results_dir(config, benchmark_path)
        
        print(f"Results will be saved to {run_path}")

        problems, outputs = utils.run_benchmark(llm, benchmark_path, config)

        utils.save_results(run_path, benchmark_path, problems, outputs)
        