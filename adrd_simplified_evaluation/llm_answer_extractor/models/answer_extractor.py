# models/answer_extractor.py

import re
import os
import pandas as pd
from utils.prompts import EXTRACT_ANSWER_PROMPT


class AnswerExtractor:
    def __init__(self, llm_wrapper, option_keys, options_list):
        self.llm_wrapper = llm_wrapper
        self.tokenizer = llm_wrapper.tokenizer
        self.option_keys = option_keys
        self.options_list = options_list

    @staticmethod
    def extract_option_keys(model_dfs):
        sample_options = model_dfs[list(model_dfs.keys())[0]].iloc[0]['problem']['options']
        keys = re.findall(r'\b([A-Z])\.', sample_options)
        return keys, sample_options

    def extract_letter(self, text):
        match = re.search(r'ANSWER:\s*([A-Z])', text)
        return match.group(1) if match else 'invalid'

    def extract_with_tag(self, text):
        match = re.search(r'<answer>\n(Answer: )([a-zA-Z])\n</answer>', text, re.DOTALL)
        return match.group(2).strip().upper() if match else 'invalid'

    def generate_answer(self, answer_dicts):
        messages = [
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": EXTRACT_ANSWER_PROMPT.format(
                answer=d['answer'], question=d['question'], option=d['option'])}]
            for d in answer_dicts
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False,
                continue_final_message=False
            ) for msg in messages
        ]
        completions = self.llm_wrapper.generate(prompts)
        return [c.outputs[0].text for c in completions]

    def run(self, df, name, benchmark):
        result_path = f"./extracted_results/{name}_{benchmark}.csv"
        if os.path.exists(result_path):
            print(f"Results file exists. Loading from {result_path}")
            combined_df = pd.read_csv(result_path)
            combined_df.index = combined_df.groupby('ID', sort=False).ngroup()
            return combined_df
        
        print(f"Processing {name}")
        df['prediction'] = df['generated_text'].apply(self.extract_with_tag)
        invalid_df = df[
            (df['prediction'] == 'invalid') | (~df['prediction'].isin(self.option_keys))
        ].copy().reset_index(drop=True)

        valid_df = df[~df['UNQ_ID'].isin(invalid_df['UNQ_ID'])].copy().reset_index(drop=True)

        answer_dicts = [
            {
                'answer': row['generated_text'],
                'question': row['problem']['question'],
                'option': self.options_list
            } for _, row in invalid_df.iterrows()
        ]

        extracted = self.generate_answer(answer_dicts)
        invalid_df['prediction'] = [self.extract_letter(x) for x in extracted]

        combined_df = pd.concat([valid_df, invalid_df], axis=0).sort_values(by='ID').reset_index(drop=True)
        combined_df['ground_truth'] = [p['ground_truth'] for p in combined_df.problem]
        combined_df.index = combined_df.groupby('ID', sort=False).ngroup()
        
        combined_df.to_csv(result_path)

        return combined_df
