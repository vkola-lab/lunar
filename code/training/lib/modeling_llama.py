# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import numpy as np
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    Resized,
    RandSpatialCropd,
    RandScaleCropd,
)
from utils.transforms import transformsFuncd

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

class SwinUNETRBase(torch.nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swinunetr = model

    def forward(self, x_in):
        hidden_states_out = self.swinunetr.swinViT(x_in, normalize=self.swinunetr.normalize)
        enc0 = self.swinunetr.encoder1(x_in)
        enc1 = self.swinunetr.encoder2(hidden_states_out[0])
        enc2 = self.swinunetr.encoder3(hidden_states_out[1])
        enc3 = self.swinunetr.encoder4(hidden_states_out[2])
        dec4 = self.swinunetr.encoder10(hidden_states_out[4])
        return dec4

LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaVisionModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
        k: convert input embedding to `k` d-dim vector
    """

    def __init__(self, config: LlamaConfig, k=5):
        super().__init__(config)
        # Initialize the tokenizer
        self.tokenizer = None
        
        # swinunetr model initialization
        self.img_size = 128
        swin_model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            img_size=(self.img_size,self.img_size,self.img_size),
            feature_size=48,
            use_checkpoint=False,
            use_v2=True,
        )
        self.img_model = SwinUNETRBase(swin_model)
        
        # define image transformations
        self.train_transforms = transformsFuncd("train", self.img_size, keys=["image"])
        self.val_transforms = transformsFuncd("val", self.img_size, keys=["image"])
        
        # Initialize the k for the projection layer
        self.k = k
        
        self.llama_proj = nn.Sequential(
            nn.Linear(768*4*4*4, 1024, bias=True),
            nn.Linear(1024, self.k * self.get_input_embeddings().embedding_dim, bias=True),
        )
        # self.llama_proj = nn.Linear(self.img_dim, self.get_input_embeddings().embedding_dim)

        
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        
    def prepare_prompt(self, input_ids, inputs_embeds):
        # Replace default embeddings with projected image embeddings
        inputs_embeds_modified = inputs_embeds.clone()
            
        for index in range(len(inputs_embeds_modified)):
            for token_id, emb in enumerate(inputs_embeds_modified[index]):
                if input_ids[index][token_id] in self.tokenizer.encode("<|start_of_mri|>", add_special_tokens=False):
                    if not self.tokenizer:
                        raise ValueError("Please set LlamaVisionModel's tokenizer using `model.set_tokenizer(tokenizer)`")
                    path = self.tokenizer.decode(input_ids[index][token_id+1])
                    # print(path)
                    
                    if path.endswith(".nii"):
                        transformed_image = self.val_transforms({"image" : path})['image'].unsqueeze(0).to(inputs_embeds_modified[index][token_id+1].dtype).to(inputs_embeds.device)
                        emb = self.img_model(transformed_image)
                    elif path.endswith(".npy"):
                        emb = torch.from_numpy(np.load(path, mmap_mode='r')).to(inputs_embeds_modified[index][token_id+1].dtype).to(inputs_embeds.device)
                        # print(path)
                    else:
                        if path != " and":
                            raise ValueError(f"Invalid path {path}")
                        continue
                    
                    inputs_embeds_modified[index][token_id+1:token_id+1+self.k] = self.llama_proj(emb.view(1, -1)).reshape(self.k, -1)
                
                    
        return inputs_embeds_modified

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(self, **kwargs) -> Union[Tuple, BaseModelOutputWithPast]:
        if 'inputs_embeds' not in kwargs or kwargs['inputs_embeds'] is None:
            kwargs['inputs_embeds'] = self.embed_tokens(kwargs.get('input_ids'))
        
        kwargs['inputs_embeds'] = self.prepare_prompt(kwargs.get('input_ids'), kwargs['inputs_embeds'])
        kwargs['input_ids'] = None
        return super().forward(**kwargs)


class LlamaVisionForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, k=5):
        super().__init__(config)
        self.model = LlamaVisionModel(config, k)
        
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.model.set_tokenizer(self.tokenizer)

