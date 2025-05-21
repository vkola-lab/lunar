# models/llm_interface.py

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class LLMWrapper:
    def __init__(self, config):
        self.model_id = config['model_id']
        self.n_devices = config['n_devices']
        self.max_model_len = config['max_model_len']
        self.max_new_tokens = config['max_new_tokens']

        self.llm = self._load_llm()
        self.tokenizer = self._load_tokenizer()

    def _load_llm(self):
        return LLM(
            model=self.model_id,
            tokenizer=self.model_id,
            tensor_parallel_size=self.n_devices,
            gpu_memory_utilization=0.9,
            max_model_len=self.max_model_len,
            enable_lora=False,
            distributed_executor_backend='mp',
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def generate(self, prompts, temperature=0.7, top_p=0.8, top_k=20):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=self.max_new_tokens,
        )
        return self.llm.generate(prompts=prompts, sampling_params=sampling_params)
