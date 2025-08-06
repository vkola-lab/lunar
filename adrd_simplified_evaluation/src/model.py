import json
import random
import importlib.util
import prompt_template
import torch
import gc
import contextlib
import ray

from vllm.sampling_params import GuidedDecodingParams
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
from pathlib import Path
from tqdm import tqdm
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)


def load_model(config, model_id):

    llm = LLM(
        model=model_id,
        tensor_parallel_size=config.n_gpus,
        # dtype='bfloat16', # type for model weights, will use what is specified in model config file
        # distributed_executor_backend="mp", # multiprocessing, we never need ray on SCC
        gpu_memory_utilization=config.gpu_memory_utilization,
        guided_decoding_backend=config.guided_decoding_backend,
        seed=config.llm_sampling_seed,
        enforce_eager=config.enforce_eager,
        enable_lora=config.enable_lora,
        max_model_len=config.max_model_len, 
        enable_chunked_prefill=config.enable_chunked_prefill,
        cpu_offload_gb=0,
        max_lora_rank=64
    )

    return llm


def make_sampling_parameters(config, json_schema=None):

    if json_schema is not None:
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
    else:
        guided_decoding_params = None

    sampling_params = SamplingParams(
        temperature=config.temperature,
        # stop=stop_tokens, # stop generation when you see these tokens
        guided_decoding=guided_decoding_params,
        min_p=config.min_p,
        top_p=config.top_p,
        # top_k=config.top_k,
        min_tokens=config.min_tokens,
        max_tokens=config.max_new_tokens,
        # repetition_penalty=config.repetition_penalty,
        n=config.n,  # number of sequences generated per prompt
    )

    return sampling_params

