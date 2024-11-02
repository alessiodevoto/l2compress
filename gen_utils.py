import os
import torch
from math import ceil
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from functools import partial
from cache import l2_compress, slide_kv_cache
from visualize import plot_kv_cache_norms, plot_attentions, plot_kv_cache_kurtosis, plot_token_embedding
from typing import Optional, Callable, Dict, List, Union, Tuple
from transformers import LlamaForCausalLM, Cache, DynamicCache
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def get_adjust_kv_strategy(skip_layers, sort_by, keep_ratio, prune_after, sort_metric, sort_descending, ):
    # we will store the past key values for each token in the dataset
    if isinstance(skip_layers, str):
        layers_to_skip = [int(l) for l in skip_layers.split(',')]
    else:
        assert isinstance(skip_layers, list)
        layers_to_skip = skip_layers

    if keep_ratio < 1:
        assert sort_by in ['key', 'value']
        assert sort_metric in ['norm', 'random', 'kurtosis', 'entropy', 'hoyer']
        adjust_kv_cache = partial(l2_compress,
                                  sort=sort_by,
                                  keep_ratio=keep_ratio,
                                  prune_after=prune_after,
                                  sort_by=sort_metric,
                                  descending=sort_descending,
                                  skip_layers=layers_to_skip)
    else:
        adjust_kv_cache = None
    return adjust_kv_cache


def visualise_norm_and_attention(past_key_values, chunk_tokens, num_eval_tokens, ):
    raise NotImplementedError
    # print(outputs.attentions[0].shape)
    step_dir = f"{chunk_out_dir}/eval_step_{num_eval_tokens}"
    os.makedirs(step_dir, exist_ok=True)
    plot_kv_cache_norms(past_key_values,
                        key_or_value='key',
                        layers_to_inspect=plot_layers,
                        heads_to_inspect=plot_heads,
                        out_file=f"{step_dir}/key_norms." + args.plots_format,
                        labels=chunk_tokens[:num_eval_tokens] if args.plot_labels else None
                        )
    plot_kv_cache_norms(past_key_values,
                        key_or_value='value',
                        layers_to_inspect=plot_layers,
                        heads_to_inspect=plot_heads,
                        out_file=f"{step_dir}/value_norms." + args.plots_format,
                        labels=chunk_tokens[:num_eval_tokens] if args.plot_labels else None
                        )
    plot_attentions(merge_attention_weights(attention_weights),
                    layers_to_inspect=plot_layers,
                    heads_to_inspect=plot_heads,
                    out_file=f"{step_dir}/attentions." + args.plots_format,
                    labels=chunk_tokens[:num_eval_tokens] if args.plot_labels else None
                    )
    plot_kv_cache_kurtosis(past_key_values,
                           layers_to_inspect=plot_layers,
                           heads_to_inspect=plot_heads,
                           out_file=f"{step_dir}/kurtosis." + args.plots_format,
                           labels=chunk_tokens[:num_eval_tokens] if args.plot_labels else None
                           )

    # print("attention_weights:",len(attention_weights))
    # print("attention_weights[0]:",len(attention_weights[0]))
    # print("attention_weights[0][0]:",attention_weights[0][0].shape)
    # print('attention_weights[1]', len(attention_weights[1]))
    # print('attention_weights[1][0]', attention_weights[1][0].shape)

    # if it's last step, plot the some of the token embeddings
    if num_eval_tokens == steps_to_visualize[-1]:
        tokens_to_plot = [int(i) for i in args.plot_rand_token_emb.split(',')]
        for position in tokens_to_plot:
            plot_token_embedding(past_key_values,
                                 layers_to_inspect=plot_layers,
                                 heads_to_inspect=plot_heads,
                                 out_file=f"{step_dir}/token_{position}_embedding." + args.plots_format,
                                 token_idx=position)


@torch.no_grad()
def generate_with_reference(
        model, tokenizer, input_ids, adjust_kv_cache: Optional[Callable], device, max_context_length,
        # visualisation parameters
        do_visualisation=False, plot_labels=None, plots_every=None, steps_to_visualize=None
):
    model.eval()
    past_key_values = None

    num_correct_tokens = 0
    nlls = []
    loss_fn = CrossEntropyLoss(reduction="none")

    # we accumulate the attention weights for each token in the dataset
    attention_weights = []

    # get all the tokens in the chunk and decode them for plotting
    # we want a list of tokens, not a tensor
    if do_visualisation:
        chunk_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist()) if plot_labels else None

    token_idx = tqdm(range(0, len(input_ids[0]) - 1))
    for cur_num_eval_tokens, idx in enumerate(token_idx):
        input_ids = input_ids[:, idx: idx + 1].to(device)

        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True
        )

        logits = outputs.logits.view(-1, model.config.vocab_size)
        past_key_values = outputs.past_key_values
        label = input_ids[:, idx + 1: idx + 2].to(logits.device).view(-1)

        # compute the negative log likelihood
        neg_log_likelihood = loss_fn(logits, label)

        # compute the next token accuracy
        predicted_next_token = torch.argmax(logits, dim=-1)
        num_correct_tokens += torch.sum(predicted_next_token == label).int().item()

        # we store the attention weights for each token, they have shape (num_layers, num_heads, seq_len, seq_len)
        # the shape changes at every iteration so we need to merge them later
        if plots_every > 0:
            attention_weights.append(outputs['attentions'])

        if adjust_kv_cache is not None and cur_num_eval_tokens:
            # we should update the kv_cache here with our method
            past_key_values = adjust_kv_cache(past_key_values)

        # if the kv cache exceeds the context length, we slide it
        past_key_values = slide_kv_cache(past_key_values, max_context_len=max_context_length)

        if do_visualisation and cur_num_eval_tokens in steps_to_visualize:
            visualise_norm_and_attention(past_key_values, chunk_tokens, cur_num_eval_tokens)

        nlls.append(neg_log_likelihood)
        token_idx.set_description(
            f"evaluated tokens: {cur_num_eval_tokens}, correct: {num_correct_tokens},  nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )


@torch.no_grad()
def compare_norm_and_attention(past_key_values, model: LlamaForCausalLM, new_token):
    # new_token: input_ids, the shape is like tensor([[id]], device='cuda')

    outputs = model(input_ids=new_token, past_key_values=past_key_values, output_attentions=True)
    attention_weights = outputs.attentions

    def layer_compare(kvs, layer_attentions):
        keys, values = kvs
        # keys: [batch_size, num_heads, seq_len, head_dims] [1, 32, seq_len, 128]
        assert keys.shape[0] == 1  # assert batch_size == 1
        sort_fn = partial(torch.norm, p=2, dim=-1)
        token_scalars = sort_fn(keys)  # [batch_size, num_heads, seq_len]
        rank_norm = token_scalars.squeeze(-1).argsort(descending=True, dim=-1)
        # lower norm means more important
        # I use descending = True because I drop tokens from the start position in the following

        layer_attentions = layer_attentions.squeeze(2)  # [batch_size, num_heads, query(1), seq_len]
        layer_attentions = layer_attentions[:, :, :-1].float()  # remove the query itself, because the query can't be dropped
        # rank_attn = layer_attentions.squeeze(-1).argsort(descending=True, dim=-1)  # higher attention means more important
        sorted_attn_scores, sorted_attn_indices = layer_attentions.sort(dim=-1, descending=False)
        # higher attn means more important

        reordered_attn_by_norm = torch.gather(layer_attentions, dim=2, index=rank_norm)
        # --> assert sorted_attn_scores == torch.gather(layer_attentions, dim=2, index=sorted_attn_indices)

        # how many proportions of attention score are dropped?
        # compare the difference
        # -- drop the tokens ordered by attention scores and L2-Norm, and compare the difference
        dropped_attn_scores_ordered_by_attn = sorted_attn_scores.cumsum(dim=2)  # the expected performance
        dropped_attn_scores_ordered_by_norm = reordered_attn_by_norm.cumsum(dim=2)
        # [bs, head, seq_len]

        # loss > 0
        diff_over_positions = dropped_attn_scores_ordered_by_norm - dropped_attn_scores_ordered_by_attn
        area = diff_over_positions.sum(-1)
        return area, dropped_attn_scores_ordered_by_attn, dropped_attn_scores_ordered_by_norm

    layers_area = []
    layers_dropped_attn_by_attn = []
    layers_dropped_attn_by_norm = []
    for layer_id in range(len(past_key_values)):
        area, dropped_attn_scores_ordered_by_attn, dropped_attn_scores_ordered_by_norm = (
            layer_compare(past_key_values[layer_id], attention_weights[layer_id]))
        layers_area.append(area)
        layers_dropped_attn_by_attn.append(dropped_attn_scores_ordered_by_attn)
        layers_dropped_attn_by_norm.append(dropped_attn_scores_ordered_by_norm)

    return layers_area, layers_dropped_attn_by_attn, layers_dropped_attn_by_norm


@torch.no_grad()
def generate_with_context(
        model: LlamaForCausalLM, tokenizer, context_ids, adjust_kv_cache: Optional[Callable],
        max_new_tokens=50, eos_token_id=None, do_sample=False, temperature=None, top_k=None, top_p=None,
        filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1,
        return_ids=False, plot_final_kv_cache_dims=False, compare_exp=False, **kwargs,
):
    model.eval()

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # step 1: encode the prompt, and generate the first token (it maybe not correct)
    outputs = model(context_ids, use_cache=True, past_key_values=None)
    past_key_values = outputs.past_key_values
    _, new_token = outputs.logits[:, -1:, :].max(dim=2)
    input_ids = new_token

    # comparing the norm based sorting and attention based sorting
    if compare_exp:
        res = compare_norm_and_attention(past_key_values, model, new_token)
        return res

    # step 2: compress the KV cache
    if adjust_kv_cache is not None:
        past_key_values = adjust_kv_cache(past_key_values)

    # step 3: generate conditioned on the compressed KV cache
    # consider to adjust KV cache during the generation
    generated_ids = [input_ids.item()]
    while True:
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True
        )
        past_key_values = outputs.past_key_values

        if do_sample:
            logits = outputs.logits[:, -1, :]

            if temperature is not None:
                logits = logits / temperature
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))  # Safety check
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits = logits.masked_fill(indices_to_remove, filter_value)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=False)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

                # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
                sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                # Keep at least min_tokens_to_keep
                sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, filter_value)

            probs = nn.functional.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
        else:
            _, new_token = outputs.logits[:, -1:, :].max(dim=2)
        input_ids = new_token
        generated_ids.append(new_token.item())
        # print(f"generate: {tokenizer.decode(generated_ids)}")
        if len(generated_ids) == max_new_tokens or generated_ids[-1] == eos_token_id:
            break

    if plot_final_kv_cache_dims:
        kv_cache_size = '\n'.join([f'layer {i} --> ' + str(kv[0].shape[2]) for i, kv in enumerate(past_key_values)])
        print(f"final kv cache size: \n {kv_cache_size}")

    if return_ids:
        return generated_ids
    else:
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response


def hf_generate_debug(model, tokenizer, inputs="what are you doing?", **kwargs):
    model.cuda()
    inputs = tokenizer(inputs, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    gen_kwargs = model.generation_config.to_dict()
    gen_kwargs.update({
        "max_new_tokens": 20
    })
    # gen_kwargs.update(kwargs)
    with torch.no_grad():
        results = model.generate(input_ids, **gen_kwargs)
        results = tokenizer.batch_decode(results)
    print(results)


def hf_chat_generate_debug(model, tokenizer, **kwargs):
    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]
    model.cuda()
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": 20
    }
    gen_kwargs.update(kwargs)
    with torch.no_grad():
        results = model.generate(input_ids, **gen_kwargs)
        results = tokenizer.batch_decode(results)
    print(results)
