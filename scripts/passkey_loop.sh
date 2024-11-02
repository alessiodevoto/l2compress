#!/bin/bash -l

set -e
set -u



SKIP_LAYERS=("0,1,2")
KEEP_RATIOS=("0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1")

#pip install transformers==4.37
MODEL="./models/llama-2-7b-longlora-32k-ft"

# MODEL="./models/llama-2-7b-80k"
pip install --upgrade transformers
for keep_ratio in "${KEEP_RATIOS[@]}"; do
  python eval_passkey.py \
    --base_model ${MODEL} \
    --keep_ratio ${keep_ratio} \
    --skip_layer "0,1,12"
done

for skip_layer in "${SKIP_LAYERS[@]}"; do
  for keep_ratio in "${KEEP_RATIOS[@]}"; do
    python eval_passkey.py \
      --base_model ${MODEL} \
      --keep_ratio ${keep_ratio} \
      --skip_layer ${skip_layer} \
      --sort_metric "random" \
      --more_suffix "random"
  done
done

for skip_layer in "${SKIP_LAYERS[@]}"; do
  for keep_ratio in "${KEEP_RATIOS[@]}"; do
    python eval_passkey.py \
      --base_model ${MODEL} \
      --keep_ratio ${keep_ratio} \
      --skip_layer ${skip_layer} \
      --sort_descending \
      --more_suffix "ascending"
  done
done

#
#pip install --upgrade transformers
#MODEL="./models/llama-2-7b-80k"
#for skip_layer in "${SKIP_LAYERS[@]}"; do
#  for keep_ratio in "${KEEP_RATIOS[@]}"; do
#    python eval_passkey.py \
#      --base_model ${MODEL} \
#      --keep_ratio ${keep_ratio} \
#      --skip_layer ${skip_layer}
#  done
#done

