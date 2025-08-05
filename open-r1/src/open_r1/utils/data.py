import logging
import pandas as pd
import datasets
from datasets import DatasetDict, concatenate_datasets, Dataset, load_dataset

from open_r1.configs import ScriptArguments
from open_r1.utils import utils
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


# def get_dataset(args: ScriptArguments) -> DatasetDict:
#     """Load a dataset or a mixture of datasets based on the configuration.

#     Args:
#         args (ScriptArguments): Script arguments containing dataset configuration.

#     Returns:
#         DatasetDict: The loaded datasets.
#     """
#     if args.dataset_name and not args.dataset_mixture:
#         logger.info(f"Loading dataset: {args.dataset_name}")
#         return datasets.load_dataset(args.dataset_name, args.dataset_config)
#     elif args.dataset_mixture:
#         logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
#         seed = args.dataset_mixture.seed
#         datasets_list = []

#         for dataset_config in args.dataset_mixture.datasets:
#             logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
#             ds = datasets.load_dataset(
#                 dataset_config.id,
#                 dataset_config.config,
#                 split=dataset_config.split,
#             )
#             if dataset_config.columns is not None:
#                 ds = ds.select_columns(dataset_config.columns)
#             if dataset_config.weight is not None:
#                 ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
#                 logger.info(
#                     f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
#                 )

#             datasets_list.append(ds)

#         if datasets_list:
#             combined_dataset = concatenate_datasets(datasets_list)
#             combined_dataset = combined_dataset.shuffle(seed=seed)
#             logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

#             if args.dataset_mixture.test_split_size is not None:
#                 combined_dataset = combined_dataset.train_test_split(
#                     test_size=args.dataset_mixture.test_split_size, seed=seed
#                 )
#                 logger.info(
#                     f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
#                 )
#                 return combined_dataset
#             else:
#                 return DatasetDict({"train": combined_dataset})
#         else:
#             raise ValueError("No datasets were loaded from the mixture configuration")

#     else:
#         raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")
    
def get_dataset(args: ScriptArguments, training_args):
# def get_dataset(dataset_path, system_prompt, n=1000000000, split=False, vision=False, train_type="grpo"):
    """
    Load the fine-tuning dataset from a json file and split into training and validation sets.
    :param config: the configurations
    :param tokenizer: tokenizer to format chat template
    :return: dictionary of training and validation datasets
    """
    
    data_df = pd.DataFrame()
    for dataset in args.dataset_name:
        if dataset.endswith("json"):
            data = utils.load_json(dataset)
            # data_df = pd.DataFrame(data)[:min(n, len(data))]
            sub_data_df = pd.DataFrame.from_dict(data, orient='index').reset_index(drop=True)
            sub_data_df = sub_data_df.rename(columns={'level_0': 'ID'})
            sub_data_df.drop("index", axis=1, inplace=True)
            # if training_args.shuffle_dataset:
            #     sub_data_df = sub_data_df.sample(frac=1, random_state=0).reset_index(drop=True)  
            
        elif dataset.endswith("csv"):
            sub_data_df = pd.read_csv(dataset).reset_index(drop=True)
            # if training_args.shuffle_dataset:
            #     sub_data_df = sub_data_df.sample(frac=1, random_state=0).reset_index(drop=True)  
            
        # elif "/" in dataset:  # likely a Hugging Face Hub name
        #     hf_data = load_dataset(dataset)
        #     if 'train' not in hf_data:
        #         raise ValueError(f"Hugging Face dataset {dataset} must have a 'train' split.")
        #     sub_data_df = hf_data['train'].to_pandas()
            
        else:
            raise ValueError(f"Invalid input file format {dataset}. Please use a `json` or a `csv` file.")
        
        print(f"Loaded {len(sub_data_df)} cases for {dataset}")
        
        data_df = pd.concat([data_df, sub_data_df], axis=0).reset_index(drop=True)
        
    if training_args.shuffle_dataset:
        print("Shuffling dataset")
        data_df = data_df.sample(frac=1, random_state=0).reset_index(drop=True)  
        
    dataset = {}

    def format_chat_template(row):
        if "grpo" in args.train_type.lower():
            row_json = [
                {"role": "system", "content": training_args.system_prompt},
                {"role": "user", "content": utils.get_template(train_type=args.train_type).format(patient=row["visit_summary"], question=row['question'], options=row['options'])},
            ]
            row["prompt"] = row_json
            
        elif "sft" in args.train_type.lower():
            row["prompt"] = [
                {"role": "user", "content": utils.get_template(train_type=args.train_type).format(patient=row["visit_summary"], question=row['question'], options=row['options'])},
            ]
            row["completion"] = [
                {"role": "assistant", "content": row["sft_answer"]},
            ]
        else:
            raise ValueError(f"Invalid train_type {args.train_type}")
        return row

    if args.data_split:
        print("Splitting data")
        dataset['train'], dataset['test'] = train_test_split(data_df, test_size=0.20, random_state=42)

        dataset['train'] = Dataset.from_pandas(dataset['train'])
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=16,
        )
        
        dataset['test'] = Dataset.from_pandas(dataset['test'])
        dataset['test'] = dataset['test'].map(
            format_chat_template,
            num_proc=16,
        )

        print("Training set size:", len(dataset['train']))
        print("Validation set size:", len(dataset['test']))
    else:
        dataset['train'] = Dataset.from_pandas(data_df)
        dataset['train'] = dataset['train'].map(
            format_chat_template,
            num_proc=16,
        )

        print("Training set size:", len(dataset['train']))
    
    return dataset