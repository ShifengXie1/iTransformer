#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=10

# Run from any directory; run.py already records training output under logs/.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
SEQ_LEN="${SEQ_LEN:-96}"
PERIOD_MODE="${PERIOD_MODE:-fixed}"
PERIOD="${PERIOD:-24}"
MIN_PERIOD="${MIN_PERIOD:-4}"
MAX_PERIOD="${MAX_PERIOD:-$SEQ_LEN}"
PERIOD_SIGMA="${PERIOD_SIGMA:-1.5}"
HARMONIC_MAX_BIN="${HARMONIC_MAX_BIN:-4}"
MAX_LAG="${MAX_LAG:-$PERIOD}"
LAG_ALPHA="${LAG_ALPHA:-0.1}"
LAG_BETA="${LAG_BETA:-0.1}"
COMPONENT_TEMPORAL_MIXER="${COMPONENT_TEMPORAL_MIXER:-conv}"
USE_ADAPTIVE_FUSION="${USE_ADAPTIVE_FUSION:-1}"

# Match decom's four horizons and d_model=d_ff=256.
# PRED_LENS="336" selects a single 336-step run.
# PERIOD_MODE=fft enables per-batch FFT period estimation.
width=256
model_name=iTransformer_pl
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192|336|720) ;;
    *) echo "Unsupported PRED_LENS entry: $pred_len" >&2; exit 1 ;;
  esac
  printf '\nRunning model=%s seq_len=%s pred_len=%s d_model=d_ff=%s period_mode=%s period=%s\n' \
    "$model_name" "$SEQ_LEN" "$pred_len" "$width" "$PERIOD_MODE" "$PERIOD"
  "$PYTHON" -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --data_path ETTh1.csv \
    --model_id "ETTh1_pl_${SEQ_LEN}_${pred_len}_w${width}" \
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
    --period_mode "$PERIOD_MODE" \
    --period "$PERIOD" \
    --min_period "$MIN_PERIOD" \
    --max_period "$MAX_PERIOD" \
    --period_sigma "$PERIOD_SIGMA" \
    --harmonic_max_bin "$HARMONIC_MAX_BIN" \
    --max_lag "$MAX_LAG" \
    --lag_alpha "$LAG_ALPHA" \
    --lag_beta "$LAG_BETA" \
    --component_temporal_mixer "$COMPONENT_TEMPORAL_MIXER" \
    --use_adaptive_fusion "$USE_ADAPTIVE_FUSION" \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --use_norm 1 \
    --des PeriodLag \
    --itr 1
done
