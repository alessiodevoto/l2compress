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
from utils import cumulative_acc, cumulative_ppl
from transformers import AutoModelForCausalLM
from cache import l2_compress
import matplotlib.pyplot as plt



HF_TOKEN = os.environ.get("HF_TOKEN", None)
DATE_FORMAT = "%m_%d_%Y-%H_%M_%S"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt=DATE_FORMAT)
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
    output_log_file = f"{experiment_dir}/exp_info.txt"
    global_stats_file = open(output_log_file, "w")
    pprint(vars(args), global_stats_file)
    
    # load the model and tokenizer
    print('Loading model:', args.model_id)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        padding_side='left',
        truncation_side="left")
    
    model = AutoModelForCausalLM.from_pretrained(
                            pretrained_model_name_or_path=args.model_id, 
                            torch_dtype=torch.float16,
                            device_map= "auto",
                            attn_implementation='eager',
                            token=HF_TOKEN or args.hf_token,
                            ).eval()
    
        # we will store the nlls for each token in the dataset
    loss_fn = CrossEntropyLoss(reduction="none")

   
    # load the dataset and split it into chunks
    #wikipedia_en = load_dataset("wikipedia", "20220301.en")["train"]
    wikipedia_en = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    chunk_dataset = ChunkDataset(wikipedia_en, tokenizer, chunk_size=args.chunk_size)
    dataloader = chunk_dataset.get_dataloader(batch_size=1, num_workers=1)
    


    num_processed_chunks = 0
    chunk = next(iter(dataloader))

    print(f"Processing chunk {num_processed_chunks} with {len(chunk['input_ids'][0])} tokens")
    
    # create a directory for each chunk
    chunk_out_dir = f"{experiment_dir}/chunk_{num_processed_chunks}"
    os.makedirs(chunk_out_dir, exist_ok=True)
    chunk_log_file_path = f"{chunk_out_dir}/chunk_{num_processed_chunks}_stats.txt"
    chunk_log_file = open(chunk_log_file_path, "w")

    # reset past_key_values for each chunk
    past_key_values = None  
    num_evaluated_tokens = 0
    num_correct_tokens, nlls = [], []
   

    # iterate over the tokens in the chunk
    token_idx = tqdm(range(len(chunk["input_ids"][0]) - 1))
    for idx in token_idx:
    
        input_ids = chunk["input_ids"][:, idx: idx + 1].to(device)
        label = chunk["input_ids"][:, idx + 1: idx + 2].to(device).view(-1)
        
        with torch.inference_mode():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True)
            
            # get the logits for the next token
            logits = outputs.logits[:,-1,:].view(-1, model.config.vocab_size)
            predicted_next_token = torch.argmax(logits, dim=-1)
            neg_log_likelihood = loss_fn(logits, label)

            # cumulate the number of correct tokens
            num_correct_tokens.append(torch.sum(predicted_next_token == label).int().item())
            nlls.append(neg_log_likelihood.item())
        
        # update the past_key_values
        past_key_values = outputs.past_key_values
        if args.keep_ratio < 1.0 :
            past_key_values = l2_compress(
                                        outputs.past_key_values, 
                                        keep_ratio=args.keep_ratio,
                                        prune_after=args.prune_after,
                                        skip_layers=[int(l) for l in args.skip_layers.split(',')],
                                        )
            
            
        token_idx.set_description(
            f"evaluated tokens: {num_evaluated_tokens}, correct: {sum(num_correct_tokens)},  nll: {neg_log_likelihood.mean().item():.2f}, ppl: {torch.exp(neg_log_likelihood).mean().item():.2f}")

        num_evaluated_tokens += 1

    # write stats to file
    # calculate the perplexity and next token accuracy
    chunk_ppl = torch.exp(torch.tensor(nlls).mean())
    chunk_next_token_acc = sum(num_correct_tokens) / num_evaluated_tokens
    chunk_final_kv_cache_size = '\n'.join([f'layer {i:02} --> ' + str(kv[0].shape[2]) for i, kv in enumerate(past_key_values)])
    print(f'Final stats on {num_evaluated_tokens} eval tokens with keep ratio {args.keep_ratio} and prune after {args.prune_after}: ', file=chunk_log_file)
    print(f'Final average perplexity: {chunk_ppl.item()}', file=chunk_log_file)
    print(f'Next token accuracy: {chunk_next_token_acc}', file=chunk_log_file)
    print(f'Final kv_cache size:\n{chunk_final_kv_cache_size}', file=chunk_log_file)
    chunk_log_file.close()
    

    # save nlls and next token acc for later plotting if needed
    torch.save(nlls, f"{chunk_out_dir}/nlls.pt")
    torch.save(num_correct_tokens, f"{chunk_out_dir}/accs.pt")

    # visualize the next token accuracy and perplexity
    fig, ax = plt.subplots(2, 1, figsize=(5, 5))
    ppl_m, ppl_std = cumulative_ppl(torch.tensor(nlls).unsqueeze(0))
    acc_m, acc_std = cumulative_acc(torch.tensor(num_correct_tokens).unsqueeze(0))

    print("Perplexity: ", ppl_m.shape, ppl_std.shape)
    # ignore the first n tokens for the plots 
    n = 30
    ax[0].plot(ppl_m.log()[n:], linestyle='-', linewidth=1)
    ax[0].set_ylabel("Log PPL")
    ax[1].plot(acc_m[n:], linestyle='-', linewidth=1)
    ax[1].set_ylabel("Next Token Accuracy")
    ax[0].set_xlabel("Input Length")
    ax[1].set_xlabel("Input Length")
    plt.tight_layout()
    plt.savefig(f"{chunk_out_dir}/quick_results.png")

    print("Results saved in ", chunk_out_dir)
    


if __name__ == "__main__":
    parser = ArgumentParser()

    # model and dataset
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")

    # script will go on until either num_samples or num_eval_tokens is reached
    parser.add_argument("--chunk_size", type=int, default=2000, help="number of tokens in each chunk of the initial dataset") 
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
 
    # kv cache compression settings
    # these only make sense if keep_ratio < 1
    parser.add_argument("--keep_ratio", type=float, default=0.98, help="the ratio of tokens to keep each time we prune the KV cache")
    parser.add_argument("--prune_after", type=int, default=1000, help="prune the KV cache after this many tokens")
    parser.add_argument("--skip_layers", type=str, default="0,1", help="comma separated list of layers to skip. No KV cache compression will be applied to these layers")

    # other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")

    args = parser.parse_args()

    main(args)