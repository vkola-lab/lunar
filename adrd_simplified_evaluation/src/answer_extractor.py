import llm_interface
from pathlib import Path
from omegaconf import OmegaConf
import prompt_templates
import utils
import pandas as pd
import re


class AnswerExtractor:

    def __init__(self, llm_extractor_config_path):

        self.extractor_config = OmegaConf.load(llm_extractor_config_path)

        self.llm = llm_interface.LLMWrapper(self.extractor_config)

    def extract_boxed(self, text):

        all_matches = re.findall(r"\\boxed\{.*\b(\S)\.?\b", text)

        # finding more than one match is ambiguous, mark that as invalid too
        if len(all_matches) == 0 or len(all_matches) > 1:
            return "invalid"

        return all_matches[0].strip().upper()

    def extract_llm(self, text, options_string):

        messages = [
            [
                {
                    "role": "user",
                    "content": prompt_templates.EXTRACT_ANSWER_PROMPT.format(
                        answer=text, options=options_string
                    ),
                }
            ]
        ]

        completions = self.llm.generate(messages)

        output = completions[0].outputs[0].text

        # remove thinking part
        cleaned = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

        # extract the boxed answer
        final = self.extract_boxed(cleaned)

        return final


    def extract_from_dir(self, dir_path):

        for jsonl_file in dir_path.rglob("*_output.jsonl"):

            results_df = self.extract_from_file(jsonl_file)

            output_path = jsonl_file.parent / (jsonl_file.stem + "_processed.parquet")

            results_df.to_parquet(output_path, index=False)

    def extract_from_file(self, file_path):

        print(f"loading results from {file_path}")

        results_df = utils.load_results(file_path)

        # extract with regex
        results_df["extracted"] = results_df["generated_text"].apply(self.extract_boxed)

        # extract with LLM
        mask = results_df["extracted"] == "invalid"

        results_df["prediction"] = results_df["extracted"]

        results_df.loc[mask, "prediction"] = results_df.loc[mask, ["generated_text","options"]].apply(
            lambda row: self.extract_llm(row["generated_text"], row["options"]),
            axis=1,  # applies the function to each row
        )

        return results_df