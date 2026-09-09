#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=10
# Resolve the project root from scripts/multivariate_forecasting/ETT.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

# Historical pairs come only from the training split. Training sample indices
# enforce memory_start + seq_len + pred_len <= current_start + seq_len.
# PRED_LENS="96 192 336 720" runs all horizons. Defaults to one horizon, with both models.
# Past embeddings select neighbors; historical futures fuse with Y_base in
# prediction space. Y_pred = Y_base + gate * (Y_ret - Y_base).
# A run evaluates base (g=0), retrieval (g=1) and fused forecasts together.
# MODELS="itransformer_retrieval" skips standalone baseline training.
# RETRIEVAL_BASE_LOSS_WEIGHT=0 and RETRIEVAL_RELIABILITY_GATE=0 are ablations.
# USE_RETRIEVAL=0 or RETRIEVAL_USE_FUTURE=0: original backbone;
# RETRIEVAL_USE_GATE=0: retrieved forecast wherever history is available;
# RETRIEVAL_WEIGHTED=0: uniform Top-K. No history always uses the backbone.
PYTHON="${PYTHON:-python}"
read -r -a horizons <<< "${PRED_LENS:-96}"
read -r -a models <<< "${MODELS:-iTransformer itransformer_retrieval}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192) width=256 ;;
    336|720) width=512 ;;
    *) echo "Unsupported prediction length: $pred_len" >&2; exit 1 ;;
  esac
  for model_name in "${models[@]}"; do
    case "$model_name" in
      iTransformer|itransformer_retrieval) ;;
      *) echo "Unsupported model: $model_name" >&2; exit 1 ;;
    esac
    "$PYTHON" -u run.py \
      --is_training "${IS_TRAINING:-1}" \
      --model_id "ETTh1_VRCompare_${SEQ_LEN:-96}_${pred_len}_seed${SEED:-2023}" \
      --model "$model_name" \
      --data ETTh1 \
      --root_path "${ROOT_PATH:-./dataset/ETT-small/}" \
      --data_path ETTh1.csv \
      --features M \
      --seq_len "${SEQ_LEN:-96}" \
      --label_len "${LABEL_LEN:-48}" \
      --pred_len "$pred_len" \
      --enc_in 7 --dec_in 7 --c_out 7 \
      --e_layers 2 --n_heads 8 --d_model "${D_MODEL:-$width}" --d_ff "${D_FF:-$width}" \
      --use_retrieval "${USE_RETRIEVAL:-1}" \
      --retrieval_top_k "${RETRIEVAL_TOP_K:-8}" \
      --retrieval_temperature "${RETRIEVAL_TEMPERATURE:-0.1}" \
      --retrieval_memory_size "${RETRIEVAL_MEMORY_SIZE:-1024}" \
      --retrieval_stride "${RETRIEVAL_STRIDE:-1}" \
      --retrieval_chunk_size "${RETRIEVAL_CHUNK_SIZE:-128}" \
      --retrieval_variable_chunk_size "${RETRIEVAL_VARIABLE_CHUNK_SIZE:-32}" \
      --retrieval_use_future "${RETRIEVAL_USE_FUTURE:-1}" \
      --retrieval_use_gate "${RETRIEVAL_USE_GATE:-1}" \
      --retrieval_weighted "${RETRIEVAL_WEIGHTED:-1}" \
      --retrieval_reliability_gate "${RETRIEVAL_RELIABILITY_GATE:-1}" \
      --retrieval_base_loss_weight "${RETRIEVAL_BASE_LOSS_WEIGHT:-0.2}" \
      --retrieval_diagnostics "${RETRIEVAL_DIAGNOSTICS:-1}" \
      --dropout "${DROPOUT:-0.1}" --batch_size "${BATCH_SIZE:-32}" \
      --train_epochs "${TRAIN_EPOCHS:-10}" --patience "${PATIENCE:-3}" \
      --learning_rate "${LEARNING_RATE:-0.0001}" --lradj "${LRADJ:-type1}" \
      --use_norm "${USE_NORM:-1}" --seed "${SEED:-2023}" \
      --des ReliabilityRetrieval --itr 1
  done
done
