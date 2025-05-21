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
from transformers import AutoTokenizer
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

    prompts = []

    print("Generating prompts:")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for problem in tqdm(problems):
        
        # print(problem)

        # prompt = prompt_template.TEMPLATE.format(patient_summary=problem["patient_summary"], question=problem["question"], options=problem['options'])
        
        prompt = prompt_template.TEMPLATE.format(patient=problem["visit_summary"], question=problem["question"], options=problem['options'])
        # prompt = prompt_template.TEMPLATE.format(patient_summary=problem["visit_summary"], question=problem["question"])

        # for key in problem: # replace all placeholders
        #     print(key)
        #     if f"{{{key}}}" in prompt:
        #         prompt = prompt.format(key=key)
        #     print(prompt)

        # turn options from a dictionary into a bullet list
        # options_list = "\n".join(
        #     ["- " + k + ": " + v for k, v in problem["options"].items()]
        # )
        # prompt = prompt.replace("{{options}}", options_list)
        
        
        if 'qwen3' in model_id.lower():
            message = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_lora=config.enable_lora,
                enable_thinking=True
            )
        else:
            message = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": "<think>\nOkay"},
            ]

            text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_lora=config.enable_lora,
                # continue_final_message=True,
            )
        # print(text)
        # raise ValueError
        
        prompts.append(text)
    
    return prompts


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

    # with open(benchmark_path / "prompt_template.txt") as f:
    #     prompt_template = f.read()

    prompts = make_prompts_from_template(problems, config, model_id)

    if config.enable_lora:
        lora_request = LoRARequest("adapter", 1, config.lora_path)
    else:
        lora_request = None

    print("Processing prompts... ")
    outputs = llm.generate(
        prompts, sampling_params, lora_request=lora_request
    )
    # print(outputs)
    # problems = [p for problem in problems for p in [problem] * config.n]

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


