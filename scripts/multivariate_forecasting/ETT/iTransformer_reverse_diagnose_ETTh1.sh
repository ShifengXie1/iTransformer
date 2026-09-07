#!/usr/bin/env bash
set -euo pipefail

# Run from any directory after activating the Python environment with PyTorch.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
python_bin=${PYTHON_BIN:-python}
run_tag="$(date +%Y%m%d_%H%M%S)_$$"

# Same settings as the existing ETTh1 96 -> 336 experiment.
# Each invocation uses run.py's seed 2023. The extra reverse network consumes
# RNG state, so equal seeds do not guarantee identical training trajectories.
common_args=(
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh1.csv
  --data ETTh1
  --features M
  --seq_len 96
  --label_len 48
  --pred_len 336
  --enc_in 7
  --dec_in 7
  --c_out 7
  --d_model 512
  --d_ff 512
  --n_heads 8
  --e_layers 2
  --d_layers 1
  --dropout 0.1
  --embed timeF
  --freq h
  --activation gelu
  --factor 1
  --use_norm 1
  --batch_size 32
  --learning_rate 0.0001
  --train_epochs 10
  --patience 3
  --lradj type1
  --num_workers 10
  --gpu 0
  --itr 1
  --reverse_recon_len 96
  --des "reverse_diag_${run_tag}"
)

run_experiment() {
  local group=$1
  local model=$2
  local weight=$3
  printf '\nStarting group %s: model=%s, reverse_loss_weight=%s\n' \
    "$group" "$model" "$weight"
  "$python_bin" -u run.py "${common_args[@]}" \
    --model_id "ETTh1_96_336_diag_${group}" \
    --model "$model" \
    --reverse_loss_weight "$weight"
}

# A: original baseline; B: current cycle loss; C: reverse loss disabled.
# A ignores reverse settings. C skips reverse reconstruction during training.
# Run sequentially on the same device. Stop immediately if any run fails.
run_experiment A iTransformer 0
run_experiment B iTransformer_reverse 0.05
run_experiment C iTransformer_reverse 0

printf '\nCompleted diagnostic batch: %s\n' "$run_tag"
printf 'Logs: logs/ (three run logs; Args include diag_A, diag_B, or diag_C).\n'
printf 'Final metrics: result_long_term_forecast.txt\n'
