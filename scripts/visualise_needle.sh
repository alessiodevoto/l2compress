#!/bin/bash -l

set -e
set -u

#MODEL="./models/llama-2-7b-longlora-32k-ft"

SKIP_LAYERS=("0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4")
KEEP_RATIOS=("0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

#SKIP_LAYERS=("0")
#KEEP_RATIOS=("0.9")

#for skip_layer in "${SKIP_LAYERS[@]}"; do
#  for keep_ratio in "${KEEP_RATIOS[@]}"; do
#    python eval_needle.py \
#      --model_path "./models/llama-2-7b-80k" \
#      --keep_ratio ${keep_ratio} \
#      --skip_layer ${skip_layer} \
#      --log_dir "./output/combined_results/llama-2-7b-80k-needle-results" \
#      --only_visualise
#  done
#done

python eval_needle.py \
  --model_path "./models/llama-2-7b-longlora-32k-ft" \
  --keep_ratio 1 \
  --skip_layer 0 \
  --log_dir "./needle/results" \
  --only_visualise


#for skip_layer in "${SKIP_LAYERS[@]}"; do
#  for keep_ratio in "${KEEP_RATIOS[@]}"; do
#    python eval_needle.py \
#      --model_path "./models/llama-2-7b-longlora-32k-ft" \
#      --keep_ratio ${keep_ratio} \
#      --skip_layer ${skip_layer} \
#      --log_dir "./output/combined_results/llama-2-7b-longlora-32k-ft-needle-results" \
#      --only_visualise
#  done
#done
