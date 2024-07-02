
#%%
import os
import torch
import argparse
import pandas as pd
import torch.distributed as dist
import json
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM
# os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
PREFIX = "./data"

#%%
def get_parser():
    parser = argparse.ArgumentParser(description="Summary Generation")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="maximum new tokens")
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct", type=str, help="huggingface model name")
    parser.add_argument("--distributed", action="store_true", help="Set True for Distributed Training")

    return parser

#%%

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    args.dist_url = "env://"
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ.get("RANK"))
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        
    else:
        print('Does not support training without GPU.')
        sys.exit(1)
    
    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(seconds=2700),
    )

    torch.cuda.set_device(args.gpu)
    # os.environ['CUDA_VISIBLE_DEVICES'] = 
    args.rank = args.gpu
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def cleanup():
    dist.destroy_process_group()

def get_summary_llama(model, tokenizer, text, device, args):
    messages = [
        {"role": "system", "content": "For the given diagnostic information, your task is to act as a behavioral neurologist and summarize the diagnosis in 4 to 5 sentences using the 'present' tense. Focus on delivering a clear diagnostic summary based on the provided information, without mentioning the tests the subject underwent or the term 'UDS'. Ensure the diagnosis identifies the cognitive status, mentions any identified disorders, and specifies the primary etiological diagnosis."},
        {"role": "user", "content": text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if args.distributed:
        outputs = model.module.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    else:
        outputs = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def main(args):
    # setup(rank, world_size)
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(f"cuda:{args.rank}" if args.distributed else f"cuda")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.rank])

    all_nacc = pd.read_csv(f"{PREFIX}/nacc_with_summary.csv")
    json_name = f"{PREFIX}/nacc_with_llama_summaries_{args.rank}.json" if args.distributed else f"{PREFIX}/nacc_with_llama_summaries.json"
    
    with open(json_name, 'w', encoding='utf-8') as json_file:
        for i, row in enumerate(tqdm(all_nacc.iterrows(), total=all_nacc.shape[0])):
            if args.distributed:
                if i % args.world_size == args.rank:
                    generate = True
                else:
                    generate = False
            else:
                generate = True
                
            if generate:
                summary = get_summary_llama(model, tokenizer, row[1]['diag_SUMMARY'], device, args)
                # llama_summaries.append(summary)
                
                # Prepare data to be written as JSON
                summary_data = {
                    "index": row[0],
                    "NACCID": row[1]["NACCID"],
                    "NACCVNUM": row[1]["NACCVNUM"],
                    "NACCNMRI": row[1]["NACCNMRI"],
                    "NACCMRSA": row[1]["NACCMRSA"],
                    "NACCADC": row[1]["NACCADC"],
                    "VISITMO": row[1]["VISITMO"],
                    "VISITDAY": row[1]["VISITDAY"],
                    "VISITYR": row[1]["VISITYR"],
                    "diag_SUMMARY": row[1]["diag_SUMMARY"],
                    "diag_LLAMA_SUMMARY": summary,
                }
                
                # Write to JSON file
                json_file.write(json.dumps(summary_data, ensure_ascii=False) + "\n")  # Append each summary as a new line
                json_file.flush()

    if args.distributed:
        cleanup()
        dist.barrier()

#%%
if __name__ == "__main__":
    os.environ['HF_HOME'] = '/projectnb/vkolagrp/skowshik/.cache/'
    parser = get_parser()
    args = parser.parse_args()
    print(args.distributed)
    
    # world_size = torch.cuda.device_count()
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
    if args.distributed:
        init_distributed_mode(args)
    main(args)
