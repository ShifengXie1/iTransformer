#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Override inherited invalid OpenMP settings before importing PyTorch.
export OMP_NUM_THREADS=10

PYTHON="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
DELAY_MODE="${DELAY_MODE:-component}"
DELAY_LAGS="${DELAY_LAGS:-0,1,2,4,8,12,24}"
# Keep the relation-estimation window fixed across ablations.
DELAY_CONTEXT_LEN="${DELAY_CONTEXT_LEN:-72}"
DELAY_HIDDEN="${DELAY_HIDDEN:-32}"
DELAY_MOVING_AVG="${DELAY_MOVING_AVG:-25}"
DELAY_GATE_INIT="${DELAY_GATE_INIT:-0.02}"

# Only this model is launched. One original iTransformer backbone, width 256.
# PRED_LENS=336 selects a pilot; DELAY_MODE=off disables the correction;
# DELAY_MODE=raw uses raw relations AND values (one guide, not capacity matched).
# V2 uses observed source(T+h-lag) only: with max lag 24, direct correction
# covers h=1..24; later horizons retain the backbone forecast.
# DELAY_LAGS=0 is an exact no-evidence control, equivalent to disabling correction.
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192|336|720) ;;
    *) echo "Unsupported PRED_LENS entry: $pred_len" >&2; exit 1 ;;
  esac
  printf '\nRunning itransformer_delay v2 (observed leaders): pred_len=%s mode=%s lags=%s context=%s\n' \
    "$pred_len" "$DELAY_MODE" "$DELAY_LAGS" "$DELAY_CONTEXT_LEN"
  "$PYTHON" -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --data_path ETTh1.csv \
    --model_id "ETTh1_delayv2_96_${pred_len}_w256" \
    --model itransformer_delay \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --n_heads 8 \
    --d_model 256 \
    --d_ff 256 \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --use_norm 1 \
    --delay_mode "$DELAY_MODE" \
    --delay_lags "$DELAY_LAGS" \
    --delay_context_len "$DELAY_CONTEXT_LEN" \
    --delay_hidden "$DELAY_HIDDEN" \
    --delay_moving_avg "$DELAY_MOVING_AVG" \
    --delay_gate_init "$DELAY_GATE_INIT" \
    --des ObservedComponentDelay \
    --itr 1
done
