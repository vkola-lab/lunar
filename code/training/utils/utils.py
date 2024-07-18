# coding=utf-8

import yaml


def stop_token_list():
    """
    The stop token list for vLLM engine
    Note: You can add more stop tokens
    if you are using other LLMs that have stop tokens
    """
    stop_tokens = [
        "Question:",
    ]

    return stop_tokens


def load_config(file_name):
    """
    Load parameters and path from the YAML file
    :param file_name: the name of the YAML file
    :return config: The configuration info
    """
    fopen = open(file_name)
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config

def load_json(dataset_path):
    """
    Load the json file
    :param dataset_path: the path to the json file
    :return data: data from json
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except:
        data = []
        with open(dataset_path, "r", encoding='utf-8') as json_file:
            for line in json_file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

    return data


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # Retrieve a list of all named parameters in the model
    model_parameters = list(model.named_parameters())

    # Calculate the total number of parameters using a generator expression
    all_param = sum(p.numel() for _, p in model_parameters)

    # Calculate the total number of trainable parameters using a generator expression
    # that filters parameters which require gradients
    trainable_params = sum(p.numel() for _, p in model_parameters if p.requires_grad)

    # Print out the number of trainable parameters, total parameters,
    # and the percentage of parameters that are trainable
    # The percentage is formatted to two decimal places
    print(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


class CustomStream:
    """
    Save all the running logs
    """

    def __init__(self, filename, console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass

