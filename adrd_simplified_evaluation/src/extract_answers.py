from omegaconf import OmegaConf



if __name__ == "__main__":

    # load configuration for judge model
    cli_config = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_config.config_file)
    config = OmegaConf.merge(file_config, cli_config) 

    # load judge model (will extract answers)
    llm = model.load_model(config, model_id)