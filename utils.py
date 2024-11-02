from functools import partial
from typing import List, Literal, Optional, Tuple, Union
import torch
from torch import nn
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import transformers
import math
import logging
import re

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def pad_and_merge_single_layer(attention_weights_list):
    """
    Pad and merge a list of attention weights tensors.

    Args:
        attention_weights_list (list): A list of attention weights tensors coming from different generation steps. Each tensor has shape (batch_size, num_heads, source_len, target_len).
        Notice that source and taget len can be different for each tensor.

    Returns:
        torch.Tensor: A tensor of attention weights for all times steps. The tensor has shape (batch_size, num_heads, cumulative_source_len, max_target_len).
    """

    # max_source_seq_len = max(map(lambda x: x.shape[2], attention_weights_list))
    max_target_seq_len = max(map(lambda x: x.shape[3], attention_weights_list))

    # print('source len, target len', max_source_seq_len, max_target_seq_len)
    # print('attention_weights_list:', attention_weights_list[0].shape, attention_weights_list[1].shape)

    # pad all attention weights so that they have the same target sequence length
    padded_attention_weights = [torch.nn.functional.pad(t, (0, max_target_seq_len - t.shape[3]), 'constant', 0) for t in
                                attention_weights_list]

    # padded_attention_weights is a list of attention maps, each of shape (batch_size, num_heads, source_len, target_len)
    # we need to merge them such that the output is of shape (batch_size, num_heads, source_len, target_len) 

    # print('padded_attention_weights:', padded_attention_weights[0].shape, padded_attention_weights[1].shape)

    #  out = torch.stack(padded_attention_weights, dim=1)
    out = torch.cat(padded_attention_weights, dim=2)
    # print('out:', out.shape)
    return out


def pad_and_merge_single_layer_old(attention_weights_list):
    """
    Pad and merge a list of attention weights tensors.

    Args:
        attention_weights_list (list): A list of attention weights tensors coming from different generation steps. Each tensor has shape (batch_size, num_heads, seq_len, seq_len).
        Notice that seq_len can be different for each tensor.

    Returns:
        torch.Tensor: A padded and merged tensor of attention weights.
    """
    # let's pad the attention weights so that they have the same sequence length

    max_seq_len = max([t.size(3) for t in attention_weights_list])

    padded_attention_weights = [torch.nn.functional.pad(t, (0, max_seq_len - t.size(3)), 'constant', 0).squeeze() for t
                                in attention_weights_list]

    return torch.stack(padded_attention_weights, dim=1).unsqueeze(0)


def merge_attention_weights(attention_weights_list):
    """
    Merge attention weights from multiple layers into a single list.

    Args:
        attention_weights_list (list): A list of attention weights. Each element in the list is a list of tensors representing the attention weights for each layer.
        [[l1, l2, l3], [l1, l2, l3], ...] where l1, l2, l3 are tensors of shape (batch_size, num_heads, seq_len, seq_len). 
        Notice that seq_len can be different for each tensor.

    Returns:
        list: A list of merged attention weights from different time steps.
    """
    num_layers = len(attention_weights_list[0])
    attn_weights_by_layer = []

    # for now we have a list of time steps, each time step has a list of attention weights for each layer

    # first, we reorganize the attention weights by layer, so that that for each layer we have a list of attention weights at different time steps

    for layer in range(num_layers):
        attn_weights_by_layer.append([])
        for aw in attention_weights_list:
            attn_weights_by_layer[layer].append(aw[layer])

    # because the sequence length can be different for each time step, we need to pad the attention weights

    return [pad_and_merge_single_layer(attention_weights) for attention_weights in attn_weights_by_layer]


def load_yukang_model(model_path, cache_dir="./cache", context_size=32768):
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)  # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        model_max_length=context_size if context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )
    model.cuda()
    model.eval()
    return model, tokenizer


def load_yaofu_model(model_path, flash_attn_2):
    from needle.replace_attention import replace_hf35
    replace_hf35()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side="left",
    )
    if flash_attn_2:
        model = LlamaForCausalLM.from_pretrained(model_path,
                                                 use_flash_attention_2=flash_attn_2,
                                                 torch_dtype=torch.bfloat16,
                                                 ).eval()
    else:
        model = LlamaForCausalLM.from_pretrained(model_path,
                                                 attn_implementation="eager",
                                                 torch_dtype=torch.bfloat16,
                                                 ).eval()

    scaling_factor = 10
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=81920, device="cpu", dtype=torch.float32)
    model.eval()
    model.cuda()
    return model, tokenizer


def load_model(model_path, flash_attn_2, hf_token, model_type="llama2"):
    if "llama-2-7b-80k" in model_path:
        return load_yaofu_model(model_path, flash_attn_2)
    elif "longlora" in model_path:
        return load_yukang_model(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side='left',
        truncation_side="left",
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    if flash_attn_2:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                # attn_implementation="flash_attention_2",
                use_flash_attention_2=True,
                token=hf_token
            )
        except Exception as err:
            logger.error(err)
            logger.info("cannot use FlashAttention2")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation="eager",
                token=hf_token
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            attn_implementation="eager",
            token=hf_token
        )
    model.eval()
    model = model.cuda()
    return model, tokenizer


class PastQueriesHook:

    def __init__(self, model):
        self.model = model
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.register_forward_hook(self.store_query_values)
            layer.self_attn.q_proj.past_queries = []

    def store_query_values(self, module, input, output):
        q_len = output.size(1)
        o = output.view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2).cpu()
        module.past_queries.append(o)

    def remove_hooks(self):
        for layer in self.model.model.layers:
            layer.self_attn.q_proj._forward_hooks.clear()

    def get_past_queries(self):
        return [torch.cat(layer.self_attn.q_proj.past_queries, dim=2) for layer in self.model.model.layers]

    def clean_past_queries(self):
        for layer in self.model.model.layers:
            layer.self_attn.q_proj.past_queries = []

class PastQueriesHookLlama3:

    def __init__(self, model):
        self.model = model
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads

        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.q_proj.register_forward_hook(self.store_query_values)
            layer.self_attn.q_proj.past_queries = []

    def store_query_values(self, module, input, output):
        q_len = output.size(1)
        o = output.view(1, q_len, self.num_heads, self.head_dim).transpose(1, 2).cpu()
        module.past_queries.append(o)

    def remove_hooks(self):
        for layer in self.model.model.layers:
            layer.self_attn.q_proj._forward_hooks.clear()

    def get_past_queries(self):
        return [torch.cat(layer.self_attn.q_proj.past_queries, dim=2) for layer in self.model.model.layers]

    def clean_past_queries(self):
        for layer in self.model.model.layers:
            layer.self_attn.q_proj.past_queries = []



def cumulative_ppl(t, avg_pool_size=None):
    nlls_cumsum = t.cumsum(-1)
    nlls_cumsum /= torch.arange(1, nlls_cumsum.shape[-1] + 1, device=t.device)
    cum_ppl = torch.exp(nlls_cumsum)
    cum_ppl_mean, cum_ppl_std = cum_ppl.mean(dim=0), cum_ppl.std(dim=0) if cum_ppl.shape[0] > 1 else  torch.zeros_like(nlls_cumsum[0])
    if avg_pool_size is not None:
        # average pooling every avg_pool_size elements
        adjust = cum_ppl_mean.shape[-1] % avg_pool_size
        if adjust > 0:
            cum_ppl_mean = cum_ppl_mean[:-adjust]
            cum_ppl_std = cum_ppl_std[:-adjust]
        cum_ppl_mean = cum_ppl_mean.reshape(-1, avg_pool_size).mean(-1)
        cum_ppl_std = cum_ppl_std.reshape(-1, avg_pool_size).mean(-1)
    return cum_ppl_mean, cum_ppl_std


def cumulative_acc(t, avg_pool_size=None):
    accs_cumsum = t.cumsum(-1).float()
    accs_cumsum /= torch.arange(1, accs_cumsum.shape[-1] + 1, device=t.device)
    cum_acc_mean, cum_acc_std =  accs_cumsum.mean(dim=0), accs_cumsum.std(dim=0) if accs_cumsum.shape[0] > 1 else torch.zeros_like(accs_cumsum[0])
    if avg_pool_size is not None:
        # average pooling every avg_pool_size elements
        adjust = cum_acc_mean.shape[-1] % avg_pool_size
        if adjust > 0:
            cum_acc_mean = cum_acc_mean[:-adjust]
            cum_acc_std = cum_acc_std[:-adjust]
        cum_acc_mean = cum_acc_mean.reshape(-1, avg_pool_size).mean(-1)
        cum_acc_std = cum_acc_std.reshape(-1, avg_pool_size).mean(-1)
    return cum_acc_mean, cum_acc_std



def load_nlls_accs(directory):
    nlls = []
    accs = []
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            if re.match(r'chunk_\d+', d):
                try:
                    nlls_chunk = torch.load(os.path.join(root, d, 'nlls.pt'))
                    accs_chunk = torch.load(os.path.join(root, d, 'accs.pt'))
                    nlls.append(nlls_chunk)
                    accs.append(accs_chunk)
                except:
                    print(f'Error loading nlls and accs for {d}')
    return torch.tensor(nlls), torch.tensor(accs)





