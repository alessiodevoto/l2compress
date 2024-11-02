from functools import partial
from transformers import LlamaForCausalLM
try:
    from transformers import GemmaForCausalLM
    gemma = True
except:
    gemma = False
    pass
from typing import List, Literal, Optional, Union
from cache import l2_compress, slide_kv_cache, suppress_peak_dim, adjust_like_fastgen
import torch
from transformers import Cache


class OccamLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, 
                config, 
                sort: Optional[Literal['key', 'value']] = None, 
                keep_ratio : float = 1.0,
                sort_descending: bool = False,
                prune_after: int = 2048,
                skip_layers: List[int] = [0],
                sort_metric: Literal['norm', 'random', 'kurtosis'] = 'norm',
                discard_tokens_older_than: Optional[int] = None,
                special_tokens_ids: Optional[List[int]] = None,
                ): #TODO should we add the parameter to remove tokens that are older than context len ?
        """
        Args:
            config: the model configuration.
            sort_by: whether to sort by key or value norms. Default is None.  
            keep_ratio: the ratio of tokens to keep for each sequence. Default is 1, which means keep all tokens. ( e.g. If keep_ratio is 0.5, then we keep half of the tokens in each sequence)
            prune_after: the number of tokens after which to prune. If seq_len is less than this value, the kv_cache will not be changed by this functioin. Default is 2048.
            sorts_metric: the metric to sort by. Default is 'norm'.
            skip_layers: the layers to skip, i.e. for which we do not prune the kvcache. Default is an empty list.
            sort_descending: whether to sort in descending order. Default is False.
        """
        super().__init__(config)
        self.config = config
        self.keep_ratio = keep_ratio
        self.sort_by = sort
        self.sort_descending = sort_descending
        self.prune_after = prune_after
        self.skip_layers = skip_layers
        self.sort_metric = sort_metric
        self.discard_tokens_older_than = discard_tokens_older_than
        self.special_tokens_ids = torch.tensor(special_tokens_ids) if special_tokens_ids is not None else None
        self.done_once = False
        self.protected_tokens = []

        if discard_tokens_older_than is None:
            print('[WARNING] discard_tokens_older_than is None, this means that the cache will not be pruned to match the context length.')
        if keep_ratio == 1.0:
            print('[WARNING] keep_ratio is 1.0, this means that the cache will not be pruned. This is equivalent to using the original model.')

    # we override this one 
    # https://github.com/huggingface/transformers/blob/96eb06286b63c9c93334d507e632c175d6ba8b28/src/transformers/models/llama/modeling_llama.py#L1116
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ): 
        
        out = super().forward(input_ids=input_ids,
                              attention_mask=attention_mask,
                              position_ids=position_ids,
                              past_key_values=past_key_values,
                              inputs_embeds=inputs_embeds,
                              labels=labels,
                              use_cache=use_cache,
                              output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states,
                              return_dict=return_dict,
                              cache_position=cache_position,
                              )

        if self.keep_ratio < 1.0 :
            # past_key_values = adjust_by_norm(out.past_key_values, 
            #                             sort=self.sort_by, 
            #                             keep_ratio=self.keep_ratio,
            #                             prune_after=self.prune_after,
            #                             skip_layers=self.skip_layers,
            #                             descending=self.sort_descending,
            #                             sort_by=self.sort_metric,
            #                            )
        
            if not self.done_once:
                print('Suppressing peak dims')
                past_key_values, c = suppress_peak_dim(out.past_key_values,
                                                prune_after=self.prune_after 
                                                )
                self.done_once = c
                out.past_key_values = past_key_values

            # get indices of special tokens in the current input ids

            # check if input_ids contains any special tokens
            
            # if self.special_tokens_ids is not None:
                # print('Checking whehter input_ids contains special tokens')
                # print(input_ids)
                # print(self.special_tokens_ids)
                # special_tokens_indices = torch.isin(input_ids, self.special_tokens_ids.to(input_ids.device))
                # print(special_tokens_indices)
                # self.protected_tokens.append(special_tokens_indices.squeeze().item())

            # print('Adjusting like fastgen')
            # past_key_values = adjust_like_fastgen(
            #                                     out.past_key_values,
            #                                     special_tokens_indices=self.protected_tokens,
            #                                     keep_ratio=self.keep_ratio,
            #                                     prune_after=self.prune_after,
            #                                     )
                out.past_key_values = past_key_values
        
        if self.discard_tokens_older_than is not None:
            past_key_values = slide_kv_cache(out.past_key_values, 
                                            max_context_len=self.discard_tokens_older_than,
                                            prune_after=self.prune_after,
                                            )
            out.past_key_values = past_key_values
        
        return out
            
if gemma:
    class OccamGemmaForCausalLM(GemmaForCausalLM):
        def __init__(self, 
                    config, 
                    sort: Optional[Literal['key', 'value']] = None, 
                    keep_ratio : float = 1.0,
                    sort_descending: bool = False,
                    prune_after: int = 2048,
                    skip_layers: List[int] = [0],
                    sort_metric: Literal['norm', 'random', 'kurtosis'] = 'norm',
                    discard_tokens_older_than: Optional[int] = None,
                    ): #TODO should we add the parameter to remove tokens that are older than context len ?
            """
            Args:
                config: the model configuration.
                sort_by: whether to sort by key or value norms. Default is None.  
                keep_ratio: the ratio of tokens to keep for each sequence. Default is 1, which means keep all tokens. ( e.g. If keep_ratio is 0.5, then we keep half of the tokens in each sequence)
                prune_after: the number of tokens after which to prune. If seq_len is less than this value, the kv_cache will not be changed by this functioin. Default is 2048.
                sorts_metric: the metric to sort by. Default is 'norm'.
                skip_layers: the layers to skip, i.e. for which we do not prune the kvcache. Default is an empty list.
                sort_descending: whether to sort in descending order. Default is False.
            """
            super().__init__(config)
            self.config = config
            self.keep_ratio = keep_ratio
            self.sort_by = sort
            self.sort_descending = sort_descending
            self.prune_after = prune_after
            self.skip_layers = skip_layers
            self.sort_metric = sort_metric
            self.discard_tokens_older_than = discard_tokens_older_than

            if discard_tokens_older_than is None:
                print('[WARNING] discard_tokens_older_than is None, this means that the cache will not be pruned to match the context length.')
            if keep_ratio == 1.0:
                print('[WARNING] keep_ratio is 1.0, this means that the cache will not be pruned. This is equivalent to using the original model.')

        # we override this one 
        # https://github.com/huggingface/transformers/blob/96eb06286b63c9c93334d507e632c175d6ba8b28/src/transformers/models/llama/modeling_llama.py#L1116
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ): 
            
            out = super().forward(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,
                                labels=labels,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states,
                                return_dict=return_dict,
                                cache_position=cache_position,
                                )

            if self.keep_ratio < 1.0:
                past_key_values = l2_compress(out.past_key_values, 
                                                sort=self.sort_by, 
                                                keep_ratio=self.keep_ratio,
                                                prune_after=self.prune_after,
                                                skip_layers=self.skip_layers,
                                                descending=self.sort_descending,
                                                sort_by=self.sort_metric,
                                                )
                out.past_key_values = past_key_values
            
            if self.discard_tokens_older_than is not None:
                past_key_values = slide_kv_cache(out.past_key_values, 
                                                max_context_len=self.discard_tokens_older_than,
                                                )
                out.past_key_values = past_key_values
            
            return out
            