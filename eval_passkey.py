# Written by Yukang Chen
# Core code based on https://github.com/CStanKonrad/long_llama
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

import argparse
import os
import random

from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from numpy import random
import torch
import transformers
import math
from gen_utils import generate_with_context, get_adjust_kv_strategy
import logging
from needle.needle_utils import get_suffix, convert_model_id_to_name
import json
from utils import load_yaofu_model, load_yukang_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

group_size_ratio = 1 / 4


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="./models/llama-2-7b-longlora-32k-ft")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=True, help='whether to use flash attention 2')
    parser.add_argument('--max_tokens', type=int, default=32000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    # parser.add_argument('--visualise', action='store_true')

    # kv cache compression settings
    parser.add_argument("--sort_by", type=str, default="key", help="key or value")
    parser.add_argument("--sort_metric", type=str, default="norm", help="norm, kurtosis or random")
    # parser.add_argument("--sort_descending", type=bool, default=False, help="sort in descending order, i.e. keep the largest norms or kurtosis")
    parser.add_argument("--sort_descending", action="store_true",
                        help="sort in descending order, i.e. keep the largest norms or kurtosis")
    parser.add_argument("--keep_ratio", type=float, default=0.8)
    parser.add_argument("--prune_after", type=int, default=0)
    parser.add_argument("--skip_layers", type=str, default="0", help="comma separated list of layers to skip")
    parser.add_argument("--more_suffix", default=None, type=str)
    parser.add_argument("--hidden_drop", action="store_true")
    args = parser.parse_args()
    return args


def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, device, use_cache=False, n_garbage=60000, seed=666,
                           adjust_kv_strategy=None):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[0, 1:]  # drop BOS
    generation_output = generate_with_context(
        model, tokenizer, input_ids, adjust_kv_cache=adjust_kv_strategy, return_ids=True
    )
    model_answer = torch.tensor(generation_output[:len(answer_ids)], device="cpu")

    # generation_output = model.generate(
    #     input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
    # )
    # model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

    is_correct = (model_answer == answer_ids).all().item()
    # print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    # print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, len_token


def get_log_name(args):
    # skip_layers, keep_ratio, sort_metric, hidden_drop, more_suffix
    suffix = get_suffix(args.skip_layers, args.keep_ratio, args.sort_metric, args.hidden_drop, args.more_suffix)
    if "/" in args.base_model:
        model_name = args.base_model.split("/")[-1]
    else:
        model_name = args.base_model
    return model_name + "_" + suffix


def main(args):
    if not os.path.exists("./output/passkey_retrieval"):
        os.makedirs("./output/passkey_retrieval")
        print("makedir: ./output/passkey_retrieval")

    log_path = os.path.join("./output/passkey_retrieval", get_log_name(args))

    if os.path.exists(log_path):
        logger.info(f"Exit, log file exists: {log_path}")
        return

    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)

    if "longlora" in args.base_model:
        model, tokenizer = load_yukang_model(args.base_model)
    elif "llama-2-7b-80k" in args.base_model:
        from needle.replace_attention import replace_hf35
        replace_hf35()
        model, tokenizer = load_yaofu_model(args.base_model, True)
    else:
        raise NotImplementedError

    adjust_kv_strategy = get_adjust_kv_strategy(
        skip_layers=args.skip_layers,
        sort_by=args.sort_by,
        keep_ratio=args.keep_ratio,
        prune_after=args.prune_after,
        sort_metric=args.sort_metric,
        sort_descending=args.sort_descending,
    )
    logger.info(f"adjust_kv_strategy: {adjust_kv_strategy}")

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        passed_tests = 0
        total_tokens = 0
        for i in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval_test(
                model, tokenizer, device, use_cache=not args.flash_attn,
                n_garbage=n_garbage, seed=i, adjust_kv_strategy=adjust_kv_strategy
            )
            passed_tests += is_correct
            total_tokens += len_tokens
        avg_tokens = total_tokens // args.num_tests
        accuracy = float(passed_tests) / args.num_tests
        print("accuracy on the token length %d is %f" % (avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)
    logger.info(f"write results to {log_path}")
    json.dump(all_accuries, open(log_path, "w"), indent=4)


if __name__ == "__main__":
    args = parse_config()
    main(args)
