#!/bin/bash -l

set -e
set -u

#   --model_path "meta-llama/Llama-2-7b-hf" \
#MODEL="yaofu/llama-2-7b-80k"
#MODEL="Yukang/Llama-2-7b-longlora-32k-ft"
#MODEL="./models/llama-2-7b-longlora-32k-ft"
MODEL="./models/llama-2-7b-80k"
FORMAT="jsonline"

python eval_icl.py \
  --model_path ${MODEL} \
  --task "sst2" \
  --seeds 42 43 44 45 46 \
  --shots 0 2 4 8 16 32 64 128 256 512 1024 \
  --batch_size 1 \
  --max_length 32768 \
  --flash_attn_2 \
  --num_examples 500 \
  --template_format ${FORMAT} \
  --input_key_verb "input: " \
  --output_key_verb "output: " \
  --intra_sep "\n" \
  --inter_sep "\n" \
  --mapping_type "regular"

python eval_icl.py \
  --model_path ${MODEL} \
  --task "sst2" \
  --seeds 42 43 44 45 46 \
  --shots 0 2 4 8 16 32 64 128 256 512 1024 \
  --batch_size 1 \
  --max_length 32768 \
  --flash_attn_2 \
  --num_examples 500 \
  --template_format ${FORMAT} \
  --input_key_verb "input: " \
  --output_key_verb "output: " \
  --intra_sep "\n" \
  --inter_sep "\n" \
  --mapping_type "unrelated"

python eval_icl.py \
  --model_path ${MODEL} \
  --task "sst2" \
  --seeds 42 43 44 45 46 \
  --shots 0 2 4 8 16 32 64 128 256 512 1024 \
  --batch_size 1 \
  --max_length 32768 \
  --flash_attn_2 \
  --num_examples 500 \
  --template_format ${FORMAT} \
  --input_key_verb "input: " \
  --output_key_verb "output: " \
  --intra_sep "\n" \
  --inter_sep "\n" \
  --mapping_type "flipped"