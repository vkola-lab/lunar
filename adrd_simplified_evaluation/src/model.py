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


def load_problems(benchmark_path,config):
    # load all JSONL files from the given path,
    # put all of them together in a list of dictionaries

    jsonl_files = benchmark_path.glob("*.jsonl")

    problems = []
    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r") as f:
            for line in f.readlines():
                problems.append(json.loads(line))

    random.seed(42)
    random.shuffle(problems)
    
    return problems[:min(config.max_questions, len(problems))]


def make_prompts_from_template(problems, config, model_id):

    # problems is a list of dicts, one dict per problem
    # we expect them to have at least the 'question' and 'options' keys

    messages = []

    print(f"Generating prompts using template style {config.template_style}:")

    for problem in tqdm(problems):
        
        if "visit_summary" in problem:
            question = f'{problem["visit_summary"]}\n\n{problem["question"]}'
        else:
            question = problem["question"]
            
        if config.template_style in ["grpo", "sft"]:
            prompt = prompt_template.TEMPLATE.format(question=question, options=problem['options'])
            
        elif config.template_style == "grpo_think":
            prompt = prompt_template.TEMPLATE_THINK.format(question=question, options=problem['options'])
            
        else:
            raise ValueError(f"Invalid template style: {config.template_style}")
        
        
        if 'qwen3' in model_id.lower() or 'sft' in model_id.lower():
            message = [
                {"role": "user", "content": prompt}
            ]
        else:
            message = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
            ]
        
        messages.append(message)
    
    return messages


def run_benchmark(llm, benchmark_path, config, model_id):

    benchmark_path = Path(benchmark_path)  # ensure it's a Path object

    if config.use_guided_decoding:
        with open(benchmark_path / "answer_schema.json") as f:
            json_schema = json.load(f)
    else:
        json_schema = None

    sampling_params = make_sampling_parameters(config, json_schema=json_schema)

    # Load all JSONL files from benchmark_path into a list of dictionaries, one dict per problem
    # we expect each problem to have a a 'question', 'options', and 'answer' keys, should that be configurable?
    problems = load_problems(benchmark_path, config)

    messages = make_prompts_from_template(problems, config, model_id)

    if config.enable_lora:
        lora_request = LoRARequest("adapter", 1, config.lora_path)
    else:
        lora_request = None

    print("Processing prompts... ")
    outputs = llm.chat(
        messages, 
        sampling_params,
        lora_request=lora_request,
        # chat_template_kwargs={"enable_thinking": False},  # Set to False to strictly disable thinking
    )

    return problems, outputs


def destroy_instance(llm):
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.\n\n\n\n")


