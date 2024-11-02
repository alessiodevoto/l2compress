from typing import List
from math import ceil
import torch
from functools import partial
from transformers import DynamicCache


def l2_compress(past_key_values,
                   keep_ratio: float = 1,
                   prune_after: int = 2048,
                   skip_layers: List = [],
                   **kwargs):
    """
    Adjust the key value cache for the model.
    The function should take in the past key values and return the adjusted key values.
    Args:
        past_key_values: the past key values from the model. This is a list of tuples, where each tuple contains the key and value tensors.  
        keep_ratio: the ratio of tokens to keep for each sequence. Default is 1, which means keep all tokens. ( e.g. If keep_ratio is 0.5, then we keep half of the tokens in each sequence)
        prune_after: the number of tokens after which to prune. If seq_len is less than this value, the kv_cache will not be changed by this functioin. Default is 2048.
        skip_layers: the layers to skip, i.e. for which we do not prune the kvcache. Default is an empty list.

    Returns:
        past_key_values: the adjusted past key values.
    """

    # both key and value have shape (batch_size, num_heads, seq_len, head_dim)
    # need a list not a tuple
    past_key_values = list(past_key_values)
   
    # iterate over the past key values, should we filter out some layers here ?
    for layer, kv in enumerate(past_key_values):

        if kv[0].size(2) < prune_after:
            continue

        keys, values = kv
        token_dim = keys.shape[-1]

        tokens_to_keep = ceil(keep_ratio * keys.size(2))

        # sort kv cache by key norm
        token_norms = torch.norm(keys, p=2, dim=-1)

        # sort by norm
        sorted_indices = token_norms.squeeze(-1).argsort(dim=-1)
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, token_dim)

        # apply sort
        sorted_keys = torch.gather(keys, dim=2, index=sorted_indices_expanded)
        sorted_values = torch.gather(values, dim=2, index=sorted_indices_expanded)

        if layer not in skip_layers:
            past_key_values[layer] = (sorted_keys[:, :, :tokens_to_keep, :], sorted_values[:, :, :tokens_to_keep, :])

    return past_key_values




