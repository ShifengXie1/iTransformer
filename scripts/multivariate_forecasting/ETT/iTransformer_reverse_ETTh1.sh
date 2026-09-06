#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root so this script also works from other directories.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=iTransformer_reverse
# Override with, e.g., REVERSE_LOSS_WEIGHT=0.01 REVERSE_RECON_LEN=48 bash ...
reverse_loss_weight=${REVERSE_LOSS_WEIGHT:-0.05}
reverse_recon_len=${REVERSE_RECON_LEN:-0}

# F: history(96) -> forecast(H); G: reversed forecast(H) -> reversed history.
# Training loss = forecast MSE + reverse_loss_weight * reconstruction MSE.
# REVERSE_RECON_LEN=0 reconstructs all 96 history steps; evaluation uses F only.
for pred_len in 96 192 336 720; do
  d_model=256
  if (( pred_len >= 336 )); then
    d_model=512
  fi

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id "ETTh1_96_${pred_len}" \
    --model "$model_name" \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model "$d_model" \
    --d_ff "$d_model" \
    --reverse_loss_weight "$reverse_loss_weight" \
    --reverse_recon_len "$reverse_recon_len" \
    --itr 1
done
