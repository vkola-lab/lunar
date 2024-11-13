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
import torch.nn as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
)
from lib.modeling_llama import LlamaVisionForCausalLM
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from utils.utils import print_parameters
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, get_gptq_peft_model
from auto_gptq.utils.peft_utils import GPTQLoraConfig


def load_model(config, use_peft=True, new_model=None, torch_dtype=torch.bfloat16, vision=False):
    """
    Initialize model for training
    :param config: the YAML configuration file
    :param use_peft: set True to use LoRA adapter
    :param new_model: adapter path if loading from adapter
    :param torch_dtype: torch dtype: default torch.bfloat16
    :param vision: set True to load from LlamaVisionForCausalLM
    
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
    freeze_lm = config.get("freeze_lm")
    freeze_input_layer = config.get("freeze_input_layer")
    
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

    # Load the base model
    if not vision:
        print("Loading AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_write_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
    else:
        print("Loading LlamaVisionForCausalLM")
        model = LlamaVisionForCausalLM.from_pretrained(
            model_name,
            use_auth_token=hf_write_token,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            k=config.get('k')
        )
        model.set_tokenizer(tokenizer)
        # model.model.config._name_or_path = config.get("hub_model_id")
        model.model.config.model_type = "llamavision"
        
        print("Copying SwinUNeTR model weights")
        model_dict = torch.load(config.get("img_ckpt_path"), map_location=model.device)
        model_dict['state_dict'] = {k.replace("module.model.",""):v for k,v in model_dict['state_dict'].items() if "module.model." in k}
        model.model.img_model.swinunetr.load_state_dict(model_dict["state_dict"])
        
        # print(model.model.llama_proj[0].weight)
        # raise ValueError
    
    # Add special tokens
    if vision:
        special_tokens_dict = {'additional_special_tokens': ['<|start_of_mri|>', '<|end_of_mri|>', '<|reserved_mri_token|>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        
        print(len(tokenizer))
        
    if freeze_lm:
        for param in model.parameters():
            param.requires_grad = False
    else:
        if use_peft:   
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
                ],
                task_type=TaskType.CAUSAL_LM,
            )
            # model = get_peft_model(model, peft_config, adapter_name="adapter")
            model.add_adapter(peft_config, adapter_name="adapter")
            model.enable_adapters()
            # print(model)
        
    
    if hasattr(model.model, 'llama_proj'):
        for param in model.model.llama_proj.parameters():
            if freeze_input_layer:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # if not freeze_input_layer:
    #     print("Enabling gradients of input embeddings.")
    #     model.model.get_input_embeddings().weight.requires_grad = True
            
    print_parameters(model=model)

    # Save the tokenizer
    # tokenizer.save_pretrained(save_dir)
    # print('Successfully save the tokenizer!')

    # Save the pre-trained model
    # model.save_pretrained(save_dir)
    # print('Successfully save the model!\n\n')

    # Clean the cache
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, tokenizer


def load_model_quant(config, use_peft=True, new_model=None, torch_dtype=torch.bfloat16, vision=False):
    """
    Initialize model for training
    :param config: the YAML configuration file
    :param use_peft: set True to use LoRA adapter
    :param new_model: adapter path if loading from adapter
    :param torch_dtype: torch dtype: default torch.bfloat16
    :param vision: set True to load from LlamaVisionForCausalLM
    
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
    freeze_lm = config.get("freeze_lm")
    freeze_input_layer = config.get("freeze_input_layer")
    
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

    # Load the base model
    print("Loading AutoModelForCausalLM")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_write_token,
        device_map=device_map,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        quantization_config= GPTQConfig(bits=4, use_exllama=False),
    )
    
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
        
    if freeze_lm:
        for param in model.parameters():
            param.requires_grad = False
    else:
        if use_peft:   
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=[
                    'up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'
                ],
                task_type=TaskType.CAUSAL_LM,
            )
            # model = get_peft_model(model, peft_config, adapter_name="adapter")
            model.add_adapter(peft_config, adapter_name="adapter")
            model.enable_adapters()
            # print(model)
            
    print_parameters(model=model)
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_dir)
    print('Successfully save the tokenizer!')

    # Save the pre-trained model
    model.save_pretrained(save_dir)
    print('Successfully save the model!\n\n')
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, tokenizer



def load_model_quant_gptq(config, use_peft=True, new_model=None, torch_dtype=torch.bfloat16, vision=False):
    """
    Initialize model for training
    :param config: the YAML configuration file
    :param use_peft: set True to use LoRA adapter
    :param new_model: adapter path if loading from adapter
    :param torch_dtype: torch dtype: default torch.bfloat16
    :param vision: set True to load from LlamaVisionForCausalLM
    
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
    freeze_lm = config.get("freeze_lm")
    freeze_input_layer = config.get("freeze_input_layer")
    
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True,
        use_triton=True,
        warmup_triton=False,
        trainable=True,
        quantize_config=None,
        # https://github.com/AutoGPTQ/AutoGPTQ/issues/210#issuecomment-1694269325
        # https://huggingface.co/TheBloke/Llama-2-70B-Chat-GPTQ/discussions/24
        # https://github.com/AutoGPTQ/AutoGPTQ/pull/237/commits/11afc47f7f9ab1671df4a81a9e91d6153d5d958e
        inject_fused_attention=False,
        inject_fused_mlp=False,
        disable_exllama=True
    )
    model.warmup_triton()
    
    model.model.quantize_config = model.quantize_config
    # model.train()
    
    model = prepare_model_for_kbit_training(model)
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    # https://gist.github.com/eusip/de8fadb761741b56d5d9a6232bf979ed#file-oasst-pythia-12b-05-03-2023-py-L68-L87
    # NOTE: https://github.com/lvwerra/trl/blob/a2749d9e0c96198486b788875eda3b325f76a5c8/examples/sentiment/scripts/gpt-neox-20b_peft/gpt-neo-20b_sentiment_peft.py#L181
    for param in model.parameters():
        # freeze base model's layers
        param.requires_grad = False

    if config.get("gradient_checkpointing"):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        use_auth_token=hf_read_token,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LoRA configurations
    peft_config = GPTQLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    model = get_gptq_peft_model(model, peft_config=peft_config, auto_find_all_linears=True, train_mode=True)
    model.print_trainable_parameters()
    
    return model, tokenizer

