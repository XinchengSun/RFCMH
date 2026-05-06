#!/bin/bash
set -e

GPU=0
SEED=10
BATCH_SIZE=64
TP=3
LR=1e-4
MAX_EPOCH=150
NOISY_RATIO=0.0

DATASETS=("xmedia" "INRIA-Websearch" "xmedianet")
BITS=(16 32 64 128)

for dataset in "${DATASETS[@]}"; do
  for bit in "${BITS[@]}"; do
    echo "=================================================="
    echo "Running dataset=${dataset}, bit=${bit}"
    echo "=================================================="
    python main.py \
      --dataset "${dataset}" \
      --GPU "${GPU}" \
      --seed "${SEED}" \
      --bit "${bit}" \
      --batch_size "${BATCH_SIZE}" \
      --tp "${TP}" \
      --lr "${LR}" \
      --MAX_EPOCH "${MAX_EPOCH}" \
      --noisy_ratio "${NOISY_RATIO}"
  done
done

echo "All 12 experiments finished."
