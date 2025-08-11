# import llm_interface
from pathlib import Path
from omegaconf import OmegaConf
import utils
import pandas as pd
import re


class AnswerExtractor:

    def __init__(self):

        # extractor_config_path = '/projectnb/vkolagrp/bellitti/adrd-foundation-model/adrd_simplified_evaluation/src/extractor_config.yml'

        # extractor_config = OmegaConf.load(extractor_config_path)

        # self.llm = llm_interface.LLMWrapper(extractor_config) 

        pass

    def extract_boxed(self, text):
        
        all_matches = re.findall(r'\\boxed\{.*\b(\S)\.?\b', text)

        # finding more than one match is ambiguous, mark that as invalid too
        if len(all_matches) == 0 or len(all_matches) > 1:
            return 'invalid'

        return all_matches[0].strip().upper()
    
    def extract_from_dir(self, dir_path):

        for jsonl_file in dir_path.rglob("*_output.jsonl"):

            results_df = self.extract_from_file(jsonl_file)

            output_path = jsonl_file.parent / (jsonl_file.stem + "_extracted.parquet")

            results_df.to_parquet(output_path, index=False)
    
    def extract_from_file(self, file_path):

        print(f'loading results from {file_path}')

        results_df = utils.load_results(file_path)

        results_df["extracted"] = results_df["generated_text"].apply(self.extract_boxed)

        return results_df
       
    # apply extract boxed to generated_text

    # run the LLM where the prediction is 'invalid' # pandas does not support null values for string types