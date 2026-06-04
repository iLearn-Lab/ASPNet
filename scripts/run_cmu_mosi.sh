#!/usr/bin/env bash
set -euo pipefail

for condition in a t v at av tv atv; do
  python -u train.py \
    --dataset CMU-MOSI \
    --audio-feature wav2vec-large-c-UTT \
    --text-feature deberta-large-4-UTT \
    --video-feature manet_UTT \
    --prompt-feature auto \
    --batch-size 32 \
    --epochs 300 \
    --stage_epoch 150 \
    --test_condition "${condition}" \
    --gpu 0
done
