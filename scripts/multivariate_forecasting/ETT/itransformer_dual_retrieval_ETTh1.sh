#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"

# Resolve the repository root from scripts/multivariate_forecasting/ETT.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Two-stage conservative residual-retrieval experiment:
#   A. train/select a standalone iTransformer;
#   B. freeze it, build hidden+shape local/global keys and residual memory;
#   C. calibrate variable-by-horizon-block correction strengths on validation.
# A candidate is already causal when its target ends before the query origin,
# so the default extra causal gap is zero. Validation selects gamma from
# DUAL_GAMMA_GRID, including gamma=0 as a complete base fallback.
# Neural gating, direct-future retrieval and metric learning remain ablations.
# Prefer the repository environment so this script can be launched directly.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv-retrieval/Scripts/python.exe" ]]; then
    PYTHON=".venv-retrieval/Scripts/python.exe"
  elif [[ -x ".venv-retrieval/bin/python" ]]; then
    PYTHON=".venv-retrieval/bin/python"
  else
    PYTHON="python"
  fi
fi
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
read -r -a seeds <<< "${SEEDS:-2023}"

for seed in "${seeds[@]}"; do
  for pred_len in "${horizons[@]}"; do
    case "$pred_len" in
      96|192) width=256 ;;
      336|720) width=512 ;;
      *) echo "Unsupported prediction length: $pred_len" >&2; exit 1 ;;
    esac

    "$PYTHON" -u run.py \
      --is_training 1 \
      --model_id "ETTh1_DualRetrieval_96_${pred_len}_seed${seed}" \
      --model itransformer_dual_retrieval \
      --seed "$seed" \
      --data ETTh1 \
      --root_path "${ROOT_PATH:-./dataset/ETT-small/}" \
      --data_path ETTh1.csv \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "$pred_len" \
      --enc_in 7 --dec_in 7 --c_out 7 \
      --e_layers 2 --n_heads 8 \
      --d_model "${D_MODEL:-$width}" --d_ff "${D_FF:-$width}" \
      --dropout "${DROPOUT:-0.1}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --dual_base_epochs "${DUAL_BASE_EPOCHS:-10}" \
      --dual_base_patience "${DUAL_BASE_PATIENCE:-3}" \
      --train_epochs "${RETRIEVAL_EPOCHS:-10}" \
      --patience "${RETRIEVAL_PATIENCE:-3}" \
      --learning_rate "${LEARNING_RATE:-0.0001}" \
      --dual_base_learning_rate "${DUAL_BASE_LR:-0}" \
      --dual_retrieval_learning_rate "${DUAL_RETRIEVAL_LR:-0}" \
      --dual_top_k "${DUAL_TOP_K:-32}" \
      --dual_global_top_k "${DUAL_GLOBAL_TOP_K:-32}" \
      --dual_temperature "${DUAL_TEMPERATURE:-0.1}" \
      --dual_memory_size "${DUAL_MEMORY_SIZE:-0}" \
      --dual_stride "${DUAL_STRIDE:-1}" \
      --dual_chunk_size "${DUAL_CHUNK_SIZE:-128}" \
      --dual_variable_chunk_size "${DUAL_VARIABLE_CHUNK_SIZE:-32}" \
      --dual_memory_batch_size "${DUAL_MEMORY_BATCH_SIZE:-128}" \
      --dual_search_metric "${DUAL_SEARCH_METRIC:-l2}" \
      --dual_shape_bins "${DUAL_SHAPE_BINS:-24}" \
      --dual_shape_weight "${DUAL_SHAPE_WEIGHT:-0.5}" \
      --dual_global_weight "${DUAL_GLOBAL_WEIGHT:-0.5}" \
      --dual_use_global "${DUAL_USE_GLOBAL:-1}" \
      --dual_use_future "${DUAL_USE_FUTURE:-0}" \
      --dual_use_residual "${DUAL_USE_RESIDUAL:-1}" \
      --dual_causal_gap "${DUAL_CAUSAL_GAP:-0}" \
      --dual_horizon_gate "${DUAL_HORIZON_GATE:-1}" \
      --dual_scale_residual "${DUAL_SCALE_RESIDUAL:-1}" \
      --dual_learned_gate "${DUAL_LEARNED_GATE:-0}" \
      --dual_train_metric "${DUAL_TRAIN_METRIC:-0}" \
      --dual_utility_loss_weight "${DUAL_UTILITY_LOSS_WEIGHT:-0.1}" \
      --dual_gamma_grid "${DUAL_GAMMA_GRID:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1}" \
      --dual_calibration_blocks "${DUAL_CALIBRATION_BLOCKS:-4}" \
      --dual_min_validation_gain "${DUAL_MIN_VALIDATION_GAIN:-0.001}" \
      --retrieval_diagnostics "${RETRIEVAL_DIAGNOSTICS:-1}" \
      --des "${DESCRIPTION:-CalibratedResidualRetrieval}"
  done
done
