import argparse
import os
import time
from collections import Counter
from eval_utils import (
    eval_generation_em,
    eval_generation_em_answers,
    exact_match_score,
    exact_match_score_with_multiple_candidates
)
import torch
import json
import numpy as np
from tqdm import tqdm
from transformers.models.llama import LlamaTokenizer
from transformers import LlamaForCausalLM
from torch.utils.data import DataLoader, Dataset
import logging
from icl.template import get_task
from utils import merge_attention_weights, load_model
from dataset import PromptDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s %(name)s %(lineno)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def parse_line_break(s):
    s = str(s)
    return s.replace("\\n", "\n")


def get_args():
    args = argparse.ArgumentParser()

    args.add_argument("--model_path", type=str, required=False, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--debug", action="store_true")

    args.add_argument("--task", type=str, required=False, default="sst2")
    args.add_argument("--seeds", type=int, default=[42], nargs='+')
    args.add_argument("--shots", type=int, default=[1024], nargs='+')
    args.add_argument("--save_vis_info", action="store_true")

    args.add_argument("--batch_size", type=int, required=False, default=1)
    args.add_argument("--max_length", type=int, default=4096)
    args.add_argument("--flash_attn_2", action="store_true")

    args.add_argument("--num_examples", default=100, type=int,
                      help="the number of test examples. -1 means use all")

    #  --- template args ---
    args.add_argument("--template_format", choices=["verbal", "jsonline", "markdown"], default="verbal")
    args.add_argument("--input_key_verb", default="input: ", type=parse_line_break)
    args.add_argument("--output_key_verb", default="output: ", type=parse_line_break)
    args.add_argument("--intra_sep", default="\n", type=parse_line_break)
    args.add_argument("--inter_sep", default="\n", type=parse_line_break)
    args.add_argument("--mapping_type", default="regular",
                      choices=["regular", "number", "alphabet", "unrelated", "flipped", "self-define"],
                      help="the type of ICL, which defines the output mapping. It is defined in template.py")

    return args.parse_args()


def normalise_pred(pred):
    return pred.strip().split("\n")[0].strip()


@torch.no_grad()
def greedy_decoding(model, tokenizer: LlamaTokenizer, model_inputs, generation_kwargs,
                    return_visualisation_info=False):
    input_ids = model_inputs["input_ids"]
    if len(input_ids) > 1:
        raise NotImplementedError("batch_size > 1 is not implemented")
    if "eos_token_id" not in generation_kwargs:
        logger.warning("eos_token_id is not set")
    max_new_tokens = generation_kwargs["max_new_tokens"]
    eos_token_id = generation_kwargs.get("eos_token_id", None)
    past_key_values = None
    generated_ids = []
    attention_weights = []
    while True:
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True
        )
        past_key_values = outputs.past_key_values
        attention_weights.append(outputs.attentions)
        _, new_token = outputs.logits[:, -1:, :].max(dim=2)
        input_ids = new_token
        generated_ids.append(new_token.item())
        if len(generated_ids) == max_new_tokens or generated_ids[-1] == eos_token_id:
            break

    gen_results = {
        "generated_ids": generated_ids,
        "generated_str": tokenizer.decode(generated_ids)
    }

    if return_visualisation_info:
        generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        prompt_tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0].tolist())
        attention_weights = merge_attention_weights(attention_weights)
        gen_results.update({
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "attention_weights": attention_weights,
            "past_key_values": past_key_values,
        })

    return gen_results


@torch.no_grad()
def generate_eval(model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed,
                  template_format, save_vis_info, save_vis_info_path):
    predictions = []
    targets = []
    bar = tqdm(total=len(prompt_list), desc=f"{task.task}-{n_shot}-{seed}")
    prompt_dataset = PromptDataset(prompt_list, tokenizer)
    dataloader = prompt_dataset.get_dataloader(batch_size, max_length)

    for bid, batch in enumerate(dataloader):
        model_inputs = batch["inputs"].to("cuda")

        if save_vis_info and bid == 0:
            gen_results = greedy_decoding(model, tokenizer, model_inputs, generation_kwargs,
                                          return_visualisation_info=save_vis_info and bid == 0)
            pred_ids = [gen_results["generated_ids"]]
            pred = [gen_results["generated_str"]]
        else:
            generate_ids = model.generate(**model_inputs, **generation_kwargs)
            pred_ids = generate_ids[:, model_inputs["input_ids"].shape[1]:]
            pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        if template_format == "jsonline":
            pred = task.process_jsonline_output(model_inputs["input_ids"], pred_ids, tokenizer)
        predictions.extend(pred)
        targets.extend(batch["targets"])
        bar.update(len(model_inputs["input_ids"]))

        if bid == 0:
            print("predict results of the first batch:")
            for cur_input, cur_pred, cur_target in zip(model_inputs["input_ids"], pred, batch["targets"]):
                print("-" * 50)
                print(tokenizer.decode(cur_input), cur_pred)
                print("target:", cur_target)
                print("-" * 50)

            if save_vis_info:
                logger.info("save visualisation information of the first example")
                torch.save(gen_results, save_vis_info_path)

    assert len(predictions) == len(targets)
    return predictions, targets


def get_sampled_demonstrations(train_data, n_shot, rng):
    rng.shuffle(train_data)
    return train_data[:n_shot]


def evaluation(model, tokenizer, generation_kwargs, task, n_shot, seed, max_length, batch_size, template_format,
               num_examples, save_vis_info, save_vis_info_path):
    generation_kwargs["max_new_tokens"] = 3  # todo: calculate max_new_token by the output space
    if template_format == "jsonline":
        generation_kwargs["max_new_tokens"] = 8
    # generation_kwargs["eos_token_id"] = tokenizer.tokenize(task.inter_sep)[-1]
    generation_kwargs["eos_token_id"] = 13  # todo: we may not use \n to sep

    rng = np.random.RandomState(seed)
    prompt_list = []
    for item in task.test_data[:num_examples]:
        demonstrations = get_sampled_demonstrations(task.train_data, n_shot, rng)
        if template_format == "verbal":
            cur_prompt = task.verbal_prompt(item, demonstrations)
        elif template_format == "jsonline":
            cur_prompt = task.jsonline_prompt(item, demonstrations)
        else:
            raise NotImplemented  # todo: markdown format

        prompt_list.append({
            "prompt": cur_prompt,
            "target": task.output_mapping(item[task.output_key])
        })

        if len(prompt_list) == 1:
            print("the first example:")
            print(json.dumps(prompt_list[0], indent=4))

    predictions, targets = generate_eval(
        model, tokenizer, prompt_list, generation_kwargs, max_length, batch_size, task, n_shot, seed,
        template_format, save_vis_info, save_vis_info_path
    )
    predictions = [normalise_pred(pred) for pred in predictions]
    score = metric_acc(predictions, targets)
    return {"score": score, "predictions": predictions, "targets": targets}


def metric_acc(predictions, targets, normalise_pred=None, normalise_target=None):
    correct_cnt = 0
    assert len(predictions) == len(targets)
    for pred, target in zip(predictions, targets):
        if normalise_pred is not None:
            pred = normalise_pred(pred)
        if normalise_target is not None:
            target = normalise_target(target)
        if target == pred:
            correct_cnt += 1
    acc = correct_cnt / len(predictions) * 100
    return acc


def main():
    args = get_args()
    logger.info(json.dumps(vars(args), indent=4))

    output_dir = os.path.join("./output/icl", time.strftime("%m_%d_%Y-%H_%M_%S"))
    if not args.debug:
        os.makedirs(output_dir)

    model, tokenizer = load_model(
        args.model_path, args.flash_attn_2, os.environ.get("HF_TOKEN", None), model_type="llama2",
    )

    task_kwargs = {
        "input_key_verb": args.input_key_verb,
        "output_key_verb": args.output_key_verb,
        "intra_sep": args.intra_sep,
        "inter_sep": args.inter_sep,
        "mapping_type": args.mapping_type,
        "output_mapping_dict": None,
    }
    task = get_task(args.task, **task_kwargs)
    # Yukang/Llama-2-7b-longlora-32k-ft

    if not args.debug:
        json.dump(vars(args), open(os.path.join(output_dir, "args.json"), "w"), indent=4, ensure_ascii=False)

    logger.info(json.dumps(vars(args), indent=4, ensure_ascii=False))

    results = []
    # evaluate with each k-shot and each seed
    for shot in args.shots:
        cur_shot_acc = []
        for seed_idx, seed in enumerate(args.seeds):
            # only save the vis info for the first seed, and then evaluation only saves the first example's vis info
            save_vis_info_path = os.path.join(output_dir, f"visualisation_example-{args.task}_shot{shot}_seed{seed}.pt")
            save_vis_info = seed_idx == 0 and args.save_vis_info
            generation_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "min_length": 1,
                "use_cache": True,
            }
            cur_res = evaluation(
                model, tokenizer, generation_kwargs, task, shot, seed,
                args.max_length - 5, args.batch_size, args.template_format, args.num_examples,
                save_vis_info, save_vis_info_path
            )  # todo: max_length - MAX_NEW_NUM_TOKENS
            score = cur_res["score"]
            logger.info(f"shot: {shot} seed: {seed} acc: {score}")
            logger.info(f"prediction counter: {Counter(cur_res['predictions'])}")
            logger.info(f"prediction counter: {Counter(cur_res['targets'])}")
            results.append({"shot": shot, "seed": seed, "score": score})
            cur_shot_acc.append(score)

        logger.info(f"shot: {shot} seed: {args.seeds} avg acc: {np.mean(cur_shot_acc)}")

    if not args.debug:
        json.dump(results, open(os.path.join(output_dir, "results.json"), "w"), indent=4, ensure_ascii=False)

    logger.info(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()
