#!/usr/bin/env bash
set -euo pipefail

# Run from any directory after activating the Python environment with PyTorch.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
python_bin=${PYTHON_BIN:-python}
run_tag="$(date +%Y%m%d_%H%M%S)_$$"
read -r -a seeds <<< "${SEEDS:-2023 2024 2025}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if (( ${#seeds[@]} == 0 )); then
  printf 'SEEDS must contain at least one integer.\n' >&2
  exit 1
fi
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    printf 'Invalid seed: %s\n' "$seed" >&2
    exit 1
  fi
done

# Verify gradients/RNG isolation/validation aggregation before full training.
"$python_bin" -m unittest discover -s tests -p test_reverse_reproducibility.py -v

# Same settings as the existing ETTh1 96 -> 336 experiment.
# Independent loader/reverse RNG streams and strict deterministic kernels make
# A/C a paired equivalence check. No AMP or multi-GPU in this diagnostic.
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
  --deterministic 1
  --diagnose_repro 1
  --des "reverse_diag_rng_${run_tag}"
)

run_experiment() {
  local group=$1
  local model=$2
  local weight=$3
  local seed=$4
  printf '\nStarting seed=%s group=%s: model=%s, reverse_loss_weight=%s\n' \
    "$seed" "$group" "$model" "$weight"
  "$python_bin" -u run.py "${common_args[@]}" \
    --model_id "ETTh1_96_336_diag_${group}" \
    --model "$model" \
    --reverse_loss_weight "$weight" \
    --seed "$seed"
}

# A: original baseline; B: current cycle loss; C: reverse loss disabled.
# A ignores reverse settings. C skips reverse reconstruction during training.
# Run sequentially on the same device. Stop immediately if any run fails.
for seed in "${seeds[@]}"; do
  run_experiment A iTransformer 0 "$seed"
  run_experiment C iTransformer_reverse 0 "$seed"
  # Stop before spending time on B if the zero-weight control is not equivalent.
  "$python_bin" utils/summarize_reverse_diagnosis.py \
    --batch-tag "$run_tag" --seed "$seed" --check-ac
  run_experiment B iTransformer_reverse 0.05 "$seed"
done

"$python_bin" utils/summarize_reverse_diagnosis.py --batch-tag "$run_tag"

printf '\nCompleted diagnostic batch: %s\n' "$run_tag"
printf 'Logs and paired summary: logs/ (three runs per seed).\n'
printf 'Final metrics: result_long_term_forecast.txt\n'
