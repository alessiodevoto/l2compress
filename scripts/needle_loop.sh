#!/bin/bash -l

set -e
set -u




#SKIP_LAYERS=("0" "0,1" "0,1,2" "0,1,12")
SKIP_LAYERS=("0,1,12")
KEEP_RATIOS=("0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

MODEL="./models/llama-2-7b-longlora-32k-ft"
for skip_layer in "${SKIP_LAYERS[@]}"; do
  for keep_ratio in "${KEEP_RATIOS[@]}"; do
    python eval_needle.py \
      --model_path ${MODEL} \
      --keep_ratio ${keep_ratio} \
      --skip_layer ${skip_layer}
  done
done

#MODEL="./models/llama-2-7b-80k"
#for skip_layer in "${SKIP_LAYERS[@]}"; do
#  for keep_ratio in "${KEEP_RATIOS[@]}"; do
#    python eval_needle.py \
#      --model_path ${MODEL} \
#      --keep_ratio ${keep_ratio} \
#      --skip_layer ${skip_layer} \
#      --sort_metric "random" \
#      --more_suffix "random"
#  done
#done
#
#pip install transformers==4.37
#MODEL="./models/llama-2-7b-longlora-32k-ft"
#for skip_layer in "${SKIP_LAYERS[@]}"; do
#  for keep_ratio in "${KEEP_RATIOS[@]}"; do
#    python eval_needle.py \
#      --model_path ${MODEL} \
#      --keep_ratio ${keep_ratio} \
#      --skip_layer ${skip_layer} \
#      --sort_metric "random" \
#      --more_suffix "random"
#  done
#done
#


#python eval_needle.py \
#  --model_path ${MODEL} \
#  --keep_ratio 0.6 \
#  --skip_layer 0,1 \
#  --sort_metric "norm"
#
#python eval_needle.py \
#  --model_path ${MODEL} \
#  --keep_ratio 0.6 \
#  --skip_layer 0,1 \
#  --sort_metric "entropy" \
#  --sort_descending
#
#python eval_needle.py \
#  --model_path ${MODEL} \
#  --keep_ratio 0.6 \
#  --skip_layer 0,1 \
#  --sort_metric "hoyer" \
#  --sort_descending
