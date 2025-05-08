from omegaconf import OmegaConf
import utils
import model

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

    print("Making directory for results... ", end="")
    run_path = utils.make_results_dir(config)
    print(run_path)

    llm = model.load_model(config)

    for benchmark in config.benchmarks:

        problems, outputs = model.run_benchmark(llm, benchmark, config)

        utils.save_results(run_path, benchmark, problems, outputs)
