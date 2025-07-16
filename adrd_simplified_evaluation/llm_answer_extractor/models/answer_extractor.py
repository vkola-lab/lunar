# models/answer_extractor.py

import re
import os
import pandas as pd
import numpy as np
from utils.prompts import EXTRACT_ANSWER_PROMPT


class AnswerExtractor:
    def __init__(self, llm_wrapper):
        self.llm_wrapper = llm_wrapper
        # self.tokenizer = llm_wrapper.tokenizer
    
    def get_invalid(self, row):
        row["keys"] = re.findall(r'\b([A-Z])\.', row['problem']["options"])
        if row['prediction'].upper() not in row['keys']:
            row["group"] = 'invalid'
        else:
            row["group"] = 'valid'
            
        return row

    def extract_letter(self, text):
        match = re.search(r'ANSWER:\s*([A-Z])', text)
        return match.group(1) if match else 'invalid'
    
    def extract_boxed(self, text):
        # match = re.search(r'<answer>\n(Answer: )([a-zA-Z])\n</answer>', text, re.DOTALL)
        match = re.search(r'.*\\boxed{(.*?)}.*', text, re.DOTALL)
        return match.group(1).strip().upper() if match else 'invalid'
    
    # def extract_answer_letter_sft(self, text):
    #     match = re.search(r'The answer is\s+([A-Za-z]):', text)
    #     return match.group(1).strip().upper() if match else 'invalid'

    # def extract_with_tag(self, text):
    #     # match = re.search(r'<answer>\n(Answer: )([a-zA-Z])\n</answer>', text, re.DOTALL)
    #     match = re.search(r'<answer>.*\s*(Answer: )([a-zA-Z]).*\s*</answer>', text, re.DOTALL)
    #     return match.group(2).strip().upper() if match else 'invalid'
    
    # def extract_with_tag_qwen3(self, text):
    #     match = re.search(r'</think>.*\s*(Answer: )([a-zA-Z]).*\s*', text, re.DOTALL)
    #     return match.group(2).strip().upper() if match else 'invalid'

    def generate_answer(self, answer_dicts, model_id):
        if 'qwen3' in model_id.lower() or 'llama' in model_id.lower():
            messages = [
                [
                    # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": EXTRACT_ANSWER_PROMPT.format(
                    answer=d['answer'], question=d['question'], option=d['option'], keys=d['keys'])}]
                for d in answer_dicts
            ]
        else:
            messages = [
                [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": EXTRACT_ANSWER_PROMPT.format(
                    answer=d['answer'], question=d['question'], option=d['option'], keys=d['keys'])}]
                for d in answer_dicts
            ]
        completions = self.llm_wrapper.generate(messages)
        return [c.outputs[0].text for c in completions]

    def run(self, df, name, benchmark, result_dir, model_id):
        result_dir = f"./{result_dir}/{benchmark}"
        result_name = f"{name}_{benchmark}.csv"
        if os.path.exists(f"{result_dir}/{result_name}"):
            print(f"Results file exists. Loading from {result_dir}/{result_name}")
            combined_df = pd.read_csv(f"{result_dir}/{result_name}")
            combined_df.index = combined_df.groupby('ID', sort=False).ngroup()
            return combined_df
        
        print(f"Processing {name}")
        # Extract answers
        # if "qwen3" in name.lower():
        #     df['prediction'] = df['generated_text'].apply(self.extract_with_tag_qwen3)
        # elif "sft" in name.lower():
        #     df['prediction'] = df['generated_text'].apply(self.extract_answer_letter_sft)
        # else:
        #     df['prediction'] = df['generated_text'].apply(self.extract_with_tag)
            
        df['prediction'] = df['generated_text'].apply(self.extract_boxed)
            
        # df = df.apply(self.extract_option_keys, axis=1)
        df = df.apply(self.get_invalid, axis=1)
        
        # Identify cases with invalid answers
        invalid_df = df[df['group'] == 'invalid'].reset_index(drop=True).copy()
        invalid_df["extraction_type"] = "llm"

        valid_df = df[~df['UNQ_ID'].isin(invalid_df['UNQ_ID'])].reset_index(drop=True).copy()
        valid_df["extraction_type"] = "regex"

        # Use LLM to extract answers for invalid cases
        answer_dicts = [
            {
                'answer': row['generated_text'],
                'question': row['problem']['question'],
                'option': row['problem']["options"],
                'keys': row['keys']
            } for _, row in invalid_df.iterrows()
        ]

        extracted = self.generate_answer(answer_dicts, model_id)
        invalid_df['prediction'] = [self.extract_letter(x) for x in extracted]
        
        invalid_df['extracted'] = extracted
        valid_df['extracted'] = np.nan

        # Combine responses
        combined_df = pd.concat([valid_df, invalid_df], axis=0).sort_values(by='ID').reset_index(drop=True)
        combined_df['ground_truth'] = [p['ground_truth'] for p in combined_df.problem]
        combined_df['ground_truth_text'] = [p['ground_truth_text'] for p in combined_df.problem]
        # combined_df['generated_text'] = [p['generated_text'] for p in combined_df.problem]
        combined_df.index = combined_df.groupby('ID', sort=False).ngroup()
        
        # Save the final dataframe
        os.makedirs(result_dir, exist_ok=True)
        combined_df.to_csv(f"{result_dir}/{result_name}", index=False)

        return combined_df
