#!/bin/bash -l

set -e
set -u

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
