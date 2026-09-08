#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=10

# Run from any directory; run.py already records training output under logs/.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
SEQ_LEN="${SEQ_LEN:-96}"
CALIBRATION_RANK="${CALIBRATION_RANK:-8}"
CALIBRATION_HIDDEN="${CALIBRATION_HIDDEN:-32}"
CALIBRATION_DROPOUT="${CALIBRATION_DROPOUT:-0.0}"
CALIBRATION_TREND_WINDOW="${CALIBRATION_TREND_WINDOW:-24}"
CALIBRATION_SCALE_LIMIT="${CALIBRATION_SCALE_LIMIT:-0.2}"
USE_CALIBRATION="${USE_CALIBRATION:-1}"
USE_HORIZON_FACTOR="${USE_HORIZON_FACTOR:-1}"
USE_CHANNEL_CONTEXT="${USE_CHANNEL_CONTEXT:-1}"
USE_GATE="${USE_GATE:-1}"

# Match the other ETTh1 scripts: four horizons and d_model=d_ff=256.
# PRED_LENS="336" selects a single run. USE_CALIBRATION=0 disables calibration.
# USE_HORIZON_FACTOR=0, USE_CHANNEL_CONTEXT=0 and USE_GATE=0 select ablations.
width=256
model_name=itransformer_calibration
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192|336|720) ;;
    *) echo "Unsupported PRED_LENS entry: $pred_len" >&2; exit 1 ;;
  esac
  printf '\nRunning model=%s seq_len=%s pred_len=%s d_model=d_ff=%s calibration=%s rank=%s\n' \
    "$model_name" "$SEQ_LEN" "$pred_len" "$width" "$USE_CALIBRATION" "$CALIBRATION_RANK"
  "$PYTHON" -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --data_path ETTh1.csv \
    --model_id "ETTh1_calib_${SEQ_LEN}_${pred_len}_w${width}" \
    --model "$model_name" \
    --data ETTh1 \
    --features M \
    --seq_len "$SEQ_LEN" \
    --label_len 48 \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --n_heads 8 \
    --d_model "$width" \
    --d_ff "$width" \
    --calibration_rank "$CALIBRATION_RANK" \
    --calibration_hidden "$CALIBRATION_HIDDEN" \
    --calibration_dropout "$CALIBRATION_DROPOUT" \
    --calibration_trend_window "$CALIBRATION_TREND_WINDOW" \
    --calibration_scale_limit "$CALIBRATION_SCALE_LIMIT" \
    --use_calibration "$USE_CALIBRATION" \
    --use_horizon_factor "$USE_HORIZON_FACTOR" \
    --use_channel_context "$USE_CHANNEL_CONTEXT" \
    --use_gate "$USE_GATE" \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --use_norm 1 \
    --des HorizonChannelCalibration \
    --itr 1
done
