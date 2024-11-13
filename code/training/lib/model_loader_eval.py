# coding=utf-8
#
# LICENSE OF THE FOLLOWING MODELS
#
# LLAMA 3 COMMUNITY LICENSE AGREEMENT
# https://llama.meta.com/llama3/license/

import gc
import torch
import wandb
import os
import trl
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from lib.modeling_llama import LlamaVisionForCausalLM
from trl import setup_chat_format
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from utils.utils import print_parameters
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
from huggingface_hub import login
from tokenizers import AddedToken
from safetensors.torch import load_file


def load_model_eval(config, base_model, lora_path=None, torch_dtype=torch.bfloat16, vision=False):
    """
    Initialize model for evaluation
    :param config: the YAML configuration file
    :param lora_path: adapter path
    :param torch_dtype: torch dtype: default torch.bfloat16
    :param vision: set True to load from `LlamaVisionForCausalLM`
    
    Returns: model and tokenizer
    """
    
    # base_model = config.get("model_name")
    device_map = config.get("device_map")
    hf_read_token = config.get("hf_read_token")
    hf_write_token = config.get("hf_write_token")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        padding_side="left",
        use_auth_token=hf_write_token,
    )
    
    if not vision:
        print("Loading AutoModelForCausalLM")
        base_model_reload = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=hf_write_token,
        )
    else:
        print("Loading LlamaVisionForCausalLM")
        base_model_reload = LlamaVisionForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=hf_write_token,
            k=config.get('k')
        )
        base_model_reload.set_tokenizer(tokenizer)
        print("Copying SwinUNeTR model weights")
        model_pth = "/projectnb/vkolagrp/projects/adrd_foundation_model/checkpoints/swinunetr/SwinUNETR_pixdim_minmaxnormalize_stripped_merged_withoutNP/model_bestVal.pt"
        model_dict = torch.load(model_pth, map_location="cuda")
        model_dict['state_dict'] = {k.replace("module.model.",""):v for k,v in model_dict['state_dict'].items() if "module.model." in k}
        base_model_reload.model.img_model.swinunetr.load_state_dict(model_dict["state_dict"])
        
    
    # device = base_model_reload.device
    # if os.path.isfile(f"{save_dir}/model_input_embeddings.pth"):
    #     embedding_input = torch.load(f"{save_dir}/model_input_embeddings.pth", map_location=device)
    #     num_embeddings = len(embedding_input['weight'])
    #     embedding_dim = len(embedding_input['weight'][0])
    #     embedding_layer = nn.Embedding(num_embeddings, embedding_dim).to(device)
    #     embedding_layer.load_state_dict(embedding_input)
    #     base_model_reload.set_input_embeddings(embedding_layer)
    #     print(f"Copied model input embeddings from {save_dir}/model_input_embeddings.pth")
    
    # if os.path.isfile(f"{save_dir}/model_output_embeddings.pth"):
    #     embedding_output = torch.load(f"{save_dir}/model_output_embeddings.pth", map_location=device)
    #     num_embeddings = len(embedding_output['weight'])
    #     embedding_dim = len(embedding_output['weight'][0])
    #     linear_layer = nn.Linear(embedding_dim, num_embeddings, bias=False).to(device)
    #     linear_layer.load_state_dict(embedding_output)
    #     base_model_reload.set_output_embeddings(linear_layer)
    #     print(f"Copied model output embeddings from {save_dir}/model_output_embeddings.pth")

    # base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
    if "llama" in base_model.lower() or "mistralai" in base_model.lower():
        tokenizer.pad_token = tokenizer.eos_token
    
    if lora_path:
        # Merge adapter with base model
        model = PeftModel.from_pretrained(base_model_reload, lora_path)
        model = model.merge_and_unload()
        
        return model, tokenizer
    return base_model_reload, tokenizer

# def load_model_eval_vllm(config, n_devices, enable_lora=True, new_model_path=None, torch_dtype="bfloat16"):
#     """
#     Initialize model for evaluation
#     :param config: the YAML configuration file
#     :param enable_lora: set True to loadd VLLM with LoRA enabled
#     :param new_model_path: set `new_model_path` if loading the merged model directly. Please set `enable_lora` to False if using `new_model_path`.
#     :param torch_dtype: torch dtype: default torch.bfloat16
    
#     Returns: model and tokenizer
#     """
#     if not new_model_path:
#         base_model = config.get("model_name")
#     else:
#         base_model = new_model_path
#     device_map = config.get("device_map")
#     hf_read_token = config.get("hf_read_token")
#     hf_write_token = config.get("hf_write_token")

#     llm = LLM(
#         model=base_model,
#         tokenizer=base_model,
#         dtype=torch_dtype,
#         tensor_parallel_size=n_devices,
#         gpu_memory_utilization=0.90,
#         enable_lora=enable_lora,
#     )
    

#     tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
    
#     # llm.model.set_tokenizer(tokenizer)

#     return llm, tokenizer


def load_model_with_llama_proj(config, tokenizer, lora_path, save_dir, torch_dtype=torch.bfloat16, push_to_hub=False, vision=True):
    """
    Initialize model for evaluation by replacing weights of `llama_proj` block with the trained weights
    NOTE: Use this only when using `LlamaVisionForCausalLM`
    :param config: the YAML configuration file
    :param lora_path: adapter path
    :param save_dir: path to load weights from
    :param torch_dtype: torch dtype: default torch.bfloat16
    
    Returns: model and tokenizer
    """
    # model, tokenizer = load_model_eval(config, base_model=config.get("model_name"), lora_path=lora_path, torch_dtype=torch_dtype, vision=True)
    base_model = config.get("model_name")
    device_map = config.get("device_map")
    hf_read_token = config.get("hf_read_token")
    hf_write_token = config.get("hf_write_token")
    
    # if lora_path:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         lora_path,
    #         use_auth_token=hf_write_token,
    #         device_map=device_map,
    #         torch_dtype=torch_dtype,
    #     )
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         base_model,
    #         use_auth_token=hf_write_token,
    #         device_map=device_map,
    #         torch_dtype=torch_dtype,
    #     )
    
    if not vision:
        print("Loading AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=hf_write_token,
        )
    else:
        print("Loading LlamaVisionForCausalLM")
        model = LlamaVisionForCausalLM.from_pretrained(
            base_model,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_auth_token=hf_write_token,
            k=config.get('k')
        )
        print("Copying SwinUNeTR model weights")
        model_pth = "/projectnb/vkolagrp/projects/adrd_foundation_model/checkpoints/swinunetr/SwinUNETR_pixdim_minmaxnormalize_stripped_merged_withoutNP/model_bestVal.pt"
        model_dict = torch.load(model_pth, map_location="cuda")
        model_dict['state_dict'] = {k.replace("module.model.",""):v for k,v in model_dict['state_dict'].items() if "module.model." in k}
        model.model.img_model.swinunetr.load_state_dict(model_dict["state_dict"])

    # base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
    if "llama" in base_model.lower() or "mistralai" in base_model.lower():
        tokenizer.pad_token = tokenizer.eos_token
    print("Embedding device:", model.model.embed_tokens.weight.device)
    
    device = model.device
    # Load llama_proj weight
    if os.path.isfile(f"{save_dir}/llama_proj.pth"):
        llama_proj = torch.load(f"{save_dir}/llama_proj.pth", map_location=device)
        for i, layer in enumerate(model.model.llama_proj):
            try:
                model.model.llama_proj[i].weight.data = llama_proj[f"{i}.weight"]
            except:
                print(f"No weight for {layer} layer")
        print(f"Copied llama_proj weights from {save_dir}/llama_proj.pth")
    
    # # Load input and output embedding weights
    if os.path.isfile(f"{save_dir}/model_input_embeddings.pth"):
        embedding_input = torch.load(f"{save_dir}/model_input_embeddings.pth", map_location=device)
        num_embeddings = len(embedding_input['weight'])
        embedding_dim = len(embedding_input['weight'][0])
        embedding_layer = nn.Embedding(num_embeddings, embedding_dim).to(device)
        embedding_layer.load_state_dict(embedding_input)
        model.set_input_embeddings(embedding_layer)
        print(f"Copied model input embeddings from {save_dir}/model_input_embeddings.pth")
        
    elif os.path.isfile(f"{save_dir}/model-00001-of-00004.safetensors"):
        file_path = f"{save_dir}/model-00001-of-00004.safetensors"
        loaded = load_file(file_path, device="cuda")
        embed_tokens = loaded['embed_tokens.weight']
        embedding_layer = nn.Embedding(embed_tokens.shape[0], embed_tokens.shape[1]).to(device)
        embedding_layer.weight.data = loaded['embed_tokens.weight']
        model.set_input_embeddings(embedding_layer)
        print(f"Copied model input embeddings from {save_dir}/model-00001-of-00004.safetensors")
    
    if os.path.isfile(f"{save_dir}/model_output_embeddings.pth"):
        embedding_output = torch.load(f"{save_dir}/model_output_embeddings.pth", map_location=device)
        num_embeddings = len(embedding_output['weight'])
        embedding_dim = len(embedding_output['weight'][0])
        linear_layer = nn.Linear(embedding_dim, num_embeddings, bias=False).to(device)
        linear_layer.load_state_dict(embedding_output)
        model.set_output_embeddings(linear_layer)
        print(f"Copied model output embeddings from {save_dir}/model_output_embeddings.pth")
        
    print(model.model.embed_tokens.weight.shape)
    print("Embedding device:", model.model.embed_tokens.weight.device)
        
    if lora_path:
        # Merge adapter with base model
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        
    if push_to_hub:
        model.config._name_or_path = config.get("hub_final_model_id")
        model.config.model_type = "llamavision"
        model.config.vocab_size = len(tokenizer)
        model.push_to_hub(config.get("hub_final_model_id"), token=config.get("hf_write_token"))
        tokenizer.push_to_hub(config.get("hub_final_model_id"), token=config.get("hf_write_token"))
        
    return model, tokenizer
    
def save_merged_model(config, lora_path, saving_path, vision=False):
    """
    Save merged model to local directory
    Initialize model for evaluation
    :param config: the YAML configuration file
    :param lora_path: adapter path
    :param saving_path: path to save it locally
    :param vision: set True to load from `LlamaVisionForCausalLM`

    Returns: model and tokenizer
    """
    model_name = config.get("model_name")
    device_map = config.get("device_map")
    hf_read_token = config.get("hf_read_token")
    hf_write_token = config.get("hf_write_token")
    save_dir = config.get("save_dir")
    lora_r = config.get("lora_r")
    lora_alpha = config.get("lora_alpha")
    lora_dropout = config.get("lora_dropout")
    cache_dir = config.get("cache_dir")
    torch_dtype = torch.bfloat16

    # Load the base model
    if not vision:
        print("Loading AutoModelForCausalLM")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_write_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
    else:
        print("Loading LlamaVisionForCausalLM")
        base_model = LlamaVisionForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_write_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding='longest',
        padding_side="right",
        truncation=True,
        return_tensors="pt",
        use_auth_token=hf_write_token,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    
    if "llama" in model_name.lower() or "mistralai" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        
    model = PeftModel.from_pretrained(base_model, lora_path, device_map=device_map)
    merged_model = model.merge_and_unload()
    
    isExist = os.path.exists(saving_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(saving_path)
        print("The saving directory is created!")
    
    merged_model.save_pretrained(saving_path, save_adapters=True, save_embedding_layers=True)
    tokenizer.save_pretrained(saving_path)