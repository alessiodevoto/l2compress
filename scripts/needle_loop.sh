#!/bin/bash -l

set -e
set -u

SKIP_LAYERS=("0,1,12")
KEEP_RATIOS=("0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

#MODEL="./models/llama-2-7b-80k"
MODEL="./models/llama-2-7b-longlora-32k-ft"
for skip_layer in "${SKIP_LAYERS[@]}"; do
  for keep_ratio in "${KEEP_RATIOS[@]}"; do
    python eval_needle.py \
      --model_path ${MODEL} \
      --keep_ratio ${keep_ratio} \
      --skip_layer ${skip_layer}
  done
done
