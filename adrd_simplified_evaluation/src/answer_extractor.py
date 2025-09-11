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
        # The pattern after the opening bracket is lazy, it finds as few character as 
        # possible until you hit a single letter (surrounded by word boundaries)
        # this means that if the output is \boxed{A and B} it will pick out A
        all_matches = re.findall(r"\\boxed\{.*?\b([A-Z0-9])\b", text)

        # finding more than one match is ambiguous, mark that as invalid too
        if len(all_matches) == 0 or len(set(all_matches)) > 1:
            return "invalid"

        return all_matches[-1].strip().upper()

    def remove_think(self, text):
        # Greedily remove all content between think tags. It removes 
        # from the first think tags to the last, even if there are more tags in between
        return re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL).strip()

    def extract_llm(self, ans_df):
        # ans_df must have 'generated_text' and 'options' as keys

        messages = [
            [
                {
                    "role": "user",
                    "content": prompt_templates.EXTRACT_ANSWER_BOXED_PROMPT.format(
                        answer=row.generated_text, options=row.options
                    ),
                }
            ]
            for row in ans_df.itertuples()
        ]

        completions = self.llm.generate(messages,enable_thinking=False)

        # remove thinking text and extract boxed answer
        # this will be a list of completions
        output = [
            self.extract_boxed(self.remove_think(completion.outputs[0].text))
            for completion in completions
        ]

        return output

    def extract_from_dir(self, dir_path):

        for jsonl_file in dir_path.rglob("*_output.jsonl"):

            # if the directory already contains a processed file, skip the directory
            if not any(jsonl_file.parent.glob('*_extracted_answers_last.parquet')):

                results_df = self.extract_from_file(jsonl_file)

                output_path = jsonl_file.parent / (jsonl_file.stem + "_extracted_answers_last.parquet")

                results_df.to_parquet(output_path, index=False)

    def extract_from_file(self, file_path):

        print(f"loading results from {file_path}")

        results_df = utils.load_results(file_path)

        # extract with regex
        results_df["extracted"] = results_df["generated_text"].apply(lambda text: self.extract_boxed(self.remove_think(text)))

        # extract with LLM
        mask = results_df["extracted"] == "invalid"

        # save invalid answers
        # results_df.loc[mask].to_json(file_path.parent / "invalid_answers.jsonl",lines=True,orient='records')

        results_df["prediction"] = results_df["extracted"]

        results_df.loc[mask, "prediction"] = self.extract_llm(
            results_df.loc[mask, ["generated_text", "options"]]
        )

        return results_df
