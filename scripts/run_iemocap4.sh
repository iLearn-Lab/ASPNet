#!/usr/bin/env bash
set -euo pipefail

for condition in a t v at av tv atv; do
  python -u train.py \
    --dataset IEMOCAP \
    --iemocap-classes 4 \
    --audio-feature wav2vec-large-c-UTT \
    --text-feature deberta-large-4-UTT \
    --video-feature manet_UTT \
    --disable-semantic-alignment \
    --batch-size 16 \
    --epochs 200 \
    --stage_epoch 100 \
    --test_condition "${condition}" \
    --gpu 0
done
