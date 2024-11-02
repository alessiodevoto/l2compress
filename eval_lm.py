from functools import partial
import re
import torch
from tqdm import tqdm
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from argparse import ArgumentParser
import logging
import time
from pprint import pprint

from dataset import ChunkDataset
from collections import defaultdict



from visualize import plot_kv_cache_norms, plot_attentions, plot_token_embedding
from utils import merge_attention_weights

from models import L2LlamaForCausalLM
try:
    from models import L2GemmaForCausalLM
except:
    pass

# evaluation script adapted from https://github.com/mit-han-lab/streaming-llm/blob/main/examples/eval_long_ppl.py


HF_TOKEN = os.environ.get("HF_TOKEN", None)
DATE_FORMAT = "%m_%d_%Y-%H_%M_%S"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt=DATE_FORMAT,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


results = {}


def main(args):
    device = args.device

    # if args.num_samples > 1: raise ValueError("Currently we only support num_samples=1")

    # logging
    # create an output directory based on date and time
    experiment_dir = os.path.join(args.output_dir, time.strftime(DATE_FORMAT))
    os.makedirs(experiment_dir, exist_ok=True)
    output_log_file = f"{experiment_dir}/exp_stats.txt"
    global_stats_file = open(output_log_file, "w")
    pprint(vars(args), global_stats_file)
    

    # when to plot the norms and attentions and which layers and heads to plot
    steps_to_visualize = list(range(args.plots_every, args.chunk_size, args.plots_every)) if args.plots_every > 0 else []
    plot_heads = [int(h) for h in args.plot_heads.split(',')] if args.plot_heads is not None else None
    plot_layers = [int(l) for l in args.plot_layers.split(',')] if args.plot_layers is not None else None
    custom_sentences = [s for s in args.custom_sentences.split(',')] if args.custom_sentences is not None else None


    # load the model and tokenizer
    print('Loading model:', args.model_id)
    YAOFU = '80k' in args.model_id

    if YAOFU:
        from needle.replace_attention import replace_hf35
        replace_hf35()
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side='left',
        truncation_side="left",
    )

    # get special tokens and bos token
    # Get special token IDs
    special_tokens = tokenizer.special_tokens_map.values()
    special_token_ids = tokenizer.convert_tokens_to_ids(list(special_tokens))
    print("Special Token IDs:", special_token_ids)

    # Get punctuation token IDs
    vocab = tokenizer.get_vocab()
    punctuation_pattern = re.compile(r'^[\.\,\!\?\:\;\-\(\)\[\]\{\}\"\'\/\\]$')
    punctuation_tokens = [token for token in vocab.keys() if punctuation_pattern.match(token)]
    punctuation_token_ids = tokenizer.convert_tokens_to_ids(punctuation_tokens)
    print("Punctuation Token IDs:", punctuation_token_ids)

    special_token_ids = special_token_ids + punctuation_token_ids
    
    model_class = L2GemmaForCausalLM if 'gemma' in args.model_id else L2LlamaForCausalLM
    
    model = model_class.from_pretrained(args.model_id, 
                                                torch_dtype=torch.float16,
                                                device_map= "auto",
                                                attn_implementation='eager',
                                                token=HF_TOKEN,
                                                sort=args.sort_by, 
                                                sort_metric=args.sort_metric, 
                                                sort_descending=args.sort_descending, 
                                                keep_ratio=args.keep_ratio, 
                                                prune_after=args.prune_after,
                                                skip_layers=[int(l) for l in args.skip_layers.split(',')],
                                                special_tokens_ids=special_token_ids,
                                                ).eval()
    
    if YAOFU:
        model.cpu()
        scaling_factor = 10
        for l in model.model.layers:
            l.self_attn.rotary_emb.scaling_factor = scaling_factor
            l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=81920, device="cpu", dtype=torch.float32)
    
    model.cuda()
   

    if custom_sentences is None:
        # load the dataset and split it into chunks
        wikipedia_en = load_dataset("wikipedia", "20220301.en")["train"]
        # wikipedia_en = load_dataset('NeelNanda/codeparrot_clean_subset_train', streaming=False, split="train")
        chunk_dataset = ChunkDataset(wikipedia_en, tokenizer, chunk_size=args.chunk_size, merge_chunks=not args.only_start, text_col='text')
        dataloader = chunk_dataset.get_dataloader(batch_size=1, num_workers=1)
    else:
        dataloader = [{'input_ids': tokenizer(quote, return_tensors='pt')['input_ids']} for quote in custom_sentences]


    # we will store the nlls for each token in the dataset
    loss_fn = CrossEntropyLoss(reduction="none")

    
    num_processed_chunks = 0
    all_chunks_ppl = []
    all_chunks_next_token_acc = []

    for chunk in dataloader:
        # only support batch_size == 1 currently
        assert len(chunk["input_ids"]) == 1, "batch_size > 1 is not supported"
        print(f"Processing chunk {num_processed_chunks} with {len(chunk['input_ids'][0])} tokens")

        if num_processed_chunks == args.num_samples:
            break
            
        # create a directory for each chunk
        chunk_out_dir = f"{experiment_dir}/chunk_{num_processed_chunks}"
        os.makedirs(chunk_out_dir, exist_ok=True)
        chunk_log_file_path = f"{chunk_out_dir}/chunk_{num_processed_chunks}_stats.txt"
        chunk_log_file = open(chunk_log_file_path, "w")

        # reset past_key_values for each chunk
        past_key_values = None  #HybridCache(config=model.config, max_batch_size=chunk["input_ids"].shape[0], max_cache_len=8000, dtype=torch.float16, device='cuda')

        num_eval_tokens = 0
        # num_correct_tokens = 0 to have cumulative acc, we use a list
        num_correct_tokens = []
        nlls = []

        # we accumulate the attention weights for each token in the dataset
        attention_weights = []

        # get all the tokens in the chunk and decode them for plotting
        # we want a list of tokens, not a tensor
        chunk_tokens = tokenizer.convert_ids_to_tokens(chunk["input_ids"][0].tolist()) if args.plot_labels else None

        token_idx = tqdm(range(args.num_prefill_tokens, len(chunk["input_ids"][0]) - 1))
        for idx in token_idx:
            
            if idx == args.num_prefill_tokens and args.num_prefill_tokens > 0:
                input_ids = chunk["input_ids"][:, :args.num_prefill_tokens].to(device)
                label = chunk["input_ids"][:, args.num_prefill_tokens: args.num_prefill_tokens + 1].to(device).view(-1)
                # print(f'Input ids 0 - {args.num_prefill_tokens}',)
            else:
                input_ids = chunk["input_ids"][:, idx: idx + 1].to(device)
                label = chunk["input_ids"][:, idx + 1: idx + 2].to(device).view(-1)
            
                # print(f'Input ids {idx} - {idx+1}', )

            with torch.no_grad():
                
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=args.plots_every > 0,
                )
                
                # always take the last token
                logits = outputs.logits[:,-1,:].view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                # print(outputs)
                
                
                # compute the negative log likelihood
                neg_log_likelihood = loss_fn(logits, label)

                # compute the next token accuracy
                # print('logits:', logits.shape)
                predicted_next_token = torch.argmax(logits, dim=-1)
                # num_correct_tokens += torch.sum(predicted_next_token == label).int().item()
                num_correct_tokens.append(torch.sum(predicted_next_token == label).int().item())
                
                # we store the attention weights for each token, they have shape (num_layers, num_heads, seq_len, seq_len)
                # the shape changes at every iteration so we need to merge them later
                if args.plots_every > 0:
                    attention_weights.append(outputs['attentions'])
                

                # if num_eval_tokens in steps_to_visualize:
                if custom_sentences is None and (num_eval_tokens in steps_to_visualize) or (num_eval_tokens >= len(chunk['input_ids'][0]) - 4) and custom_sentences:
                    # print(outputs.attentions[0].shape)
                    step_dir = f"{chunk_out_dir}/eval_step_{num_eval_tokens}"
                    os.makedirs(step_dir, exist_ok=True)
                    plot_kv_cache_norms(past_key_values, 
                                        key_or_value='key', 
                                        layers_to_inspect=plot_layers,
                                        heads_to_inspect=plot_heads, 
                                        out_file=f"{step_dir}/key_norms."+args.plots_format,
                                        labels=chunk_tokens[:num_eval_tokens] if args.plot_labels else None
                                        )
                    plot_kv_cache_norms(past_key_values, 
                                        key_or_value='value', 
                                        layers_to_inspect=plot_layers,
                                        heads_to_inspect=plot_heads, 
                                        out_file=f"{step_dir}/value_norms."+args.plots_format,
                                        labels=chunk_tokens[:num_eval_tokens] if args.plot_labels else None
                                        )
                    plot_attentions(merge_attention_weights(attention_weights), 
                                    layers_to_inspect=plot_layers,
                                    heads_to_inspect=plot_heads, 
                                    out_file=f"{step_dir}/attentions."+args.plots_format,
                                    labels=chunk_tokens[:num_eval_tokens+1] if args.plot_labels else None
                                    )

                    
                    #print("attention_weights:",len(attention_weights))
                    #print("attention_weights[0]:",len(attention_weights[0]))
                    #print("attention_weights[0][0]:",attention_weights[0][0].shape)
                    #print('attention_weights[1]', len(attention_weights[1]))
                    #print('attention_weights[1][0]', attention_weights[1][0].shape)
                    
                    # if it's last step, plot the some of the token embeddings
                    #if num_eval_tokens == steps_to_visualize[-1]:
                    tokens_to_plot = [int(i) for i in args.plot_rand_token_emb.split(',')]
                    for position in tokens_to_plot:
                        plot_token_embedding(past_key_values,
                                             key_or_value='key', 
                                                layers_to_inspect=plot_layers,
                                                heads_to_inspect=plot_heads,
                                                out_file=f"{step_dir}/token_{position}_embedding."+args.plots_format,
                                                token_idx=position,
                                                normalize=False,
                                                token_label=chunk_tokens[position] if args.plot_labels else f'token_{position}'
                                                )
                                                    

                
            nlls.append(neg_log_likelihood.item())
            token_idx.set_description(
                f"evaluated tokens: {num_eval_tokens}, correct: {sum(num_correct_tokens)},  nll: {neg_log_likelihood.mean().item():.2f}, ppl: {torch.exp(neg_log_likelihood).mean().item():.2f}"
            )

            # if we want to log per token info to file
            # print(neg_log_likelihood.item(), file=chunk_log_file, flush=True)
            

            num_eval_tokens += 1

        num_processed_chunks += 1
        

        # write final per chunk stats
        chunk_ppl = torch.exp(torch.tensor(nlls).mean())
        # chunk_next_token_acc = num_correct_tokens / num_eval_tokens
        chunk_next_token_acc = sum(num_correct_tokens) / num_eval_tokens
        
        
        chunk_final_kv_cache_size = '\n'.join([f'layer {i:02} --> ' + str(kv[0].shape[2]) for i, kv in enumerate(past_key_values)])
        print(f'Final stats on {num_eval_tokens} eval tokens with keep ratio {args.keep_ratio} and prune after {args.prune_after}: ', file=chunk_log_file)
        print(f'Final average perplexity: {chunk_ppl.item()}', file=chunk_log_file)
        print(f'Next token accuracy: {chunk_next_token_acc}', file=chunk_log_file)
        print(f'Final kv_cache size:\n{chunk_final_kv_cache_size}', file=chunk_log_file)

        # save to chunk log file
        chunk_log_file.close()
        
        # save nlls and next token acc to file
        torch.save(nlls, f"{chunk_out_dir}/nlls.pt")
        torch.save(num_correct_tokens, f"{chunk_out_dir}/accs.pt")

        # append final stats to global stats
        print(f"Chunk {num_processed_chunks-1} -> perplexity: {chunk_ppl.item()};  next token acc: {chunk_next_token_acc}", file=global_stats_file)
        all_chunks_ppl.append(chunk_ppl)
        all_chunks_next_token_acc.append(chunk_next_token_acc)

       


    # compute average final stats
    all_chunks_ppl = torch.tensor(all_chunks_ppl)
    all_chunks_next_token_acc = torch.tensor(all_chunks_next_token_acc)
    final_avg_ppl = torch.mean(all_chunks_ppl).item()
    final_ppl_std = torch.std(all_chunks_ppl).item() if len(all_chunks_ppl) > 1 else 0.
    final_avg_next_token_acc = torch.mean(all_chunks_next_token_acc).item()
    final_next_token_acc_std = torch.std(all_chunks_next_token_acc).item() if len(all_chunks_next_token_acc) > 1 else 0.
    
    # write final stats to file
    global_stats_file.write(f"Final average perplexity on all chunks: {final_avg_ppl} +/- {final_ppl_std}\n")
    global_stats_file.write(f"Final average next token acc on all chunks: {final_avg_next_token_acc} +/- {final_next_token_acc_std}\n")
    global_stats_file.close()

    # save all_chunks_ppl and all_chunks_next_token_acc to file
    results['avg_ppl'] = final_avg_ppl
    results['avg_next_token_acc'] = final_avg_next_token_acc
    results['std_ppl'] = final_ppl_std
    results['std_next_token_acc'] = final_next_token_acc_std
    torch.save(results, f"{experiment_dir}/results.pt")






if __name__ == "__main__":
    parser = ArgumentParser()

    # model and dataset
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
    # meta-llama/Llama-2-7b-hf
    # EleutherAI/pythia-70m-deduped
    # meta-llama/Meta-Llama-3-8B
    # meta-llama/Llama-2-13b-hf
    # 

    # script will go on until either num_samples or num_eval_tokens is reached
    parser.add_argument("--num_samples", type=int, default=1, help="max number of chunks to evaluate")
    parser.add_argument("--chunk_size", type=int, default=200, help="number of tokens in each chunk of the initial dataset") 
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    parser.add_argument("--num_prefill_tokens", type=int, default=0, help="number of tokens to prefill the cache with")
    parser.add_argument("--only_start", action='store_true', help="whether to merge the chunks into one dataset or to keep only the start of each chunk")
 
    # kv cache compression settings
    # these only make sense if keep_ratio < 1
    parser.add_argument("--keep_ratio", type=float, default=0.98)
    parser.add_argument("--sort_by", type=str, default="key", help="key or value")
    parser.add_argument("--sort_metric", type=str, default="norm")
    parser.add_argument("--sort_descending", action="store_true", help="sort in descending order, i.e. keep the largest norms")
    parser.add_argument("--prune_after", type=int, default=10)
    parser.add_argument("--skip_layers", type=str, default="0", help="comma separated list of layers to skip. No KV cache compression will be applied to these layers")

    # visualizations
    parser.add_argument("--plots_every", type=int, default=0, help="plot norms and attentions every n tokens. 0 to disable")
    parser.add_argument("--plot_layers", type=str, default='0,4,8,12,16,20,24,28', help="comma separated list of layers to plot")
    parser.add_argument("--plot_heads", type=str, default='0,4,8,12,16,20,24,28', help="comma separated list of heads to plot")
    parser.add_argument("--plots_format", type=str, default="png", choices=["png", "pdf", "html"])
    parser.add_argument("--plot_rand_token_emb", type=str, default="0", help="which token to plot the embedding of")   
    parser.add_argument("--plot_labels", action='store_true', help="whether to plot labels on the plots")
    parser.add_argument("--custom_sentences", type=str, default=None, help='comma separated list of sentences to plot')

    # other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    main(args)