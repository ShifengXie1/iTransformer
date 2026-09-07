#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=10

# Run from any directory; run.py already records training output under logs/.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
SEQ_LEN="${SEQ_LEN:-96}"
MOVING_AVG="${MOVING_AVG:-25}"

# Run decom at all four horizons, with width 256 in each branch.
# PRED_LENS="336" selects a single 336-step run.
width=256
model_name=iTransformer_decom
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192|336|720) ;;
    *) echo "Unsupported PRED_LENS entry: $pred_len" >&2; exit 1 ;;
  esac
  printf '\nRunning model=%s seq_len=%s pred_len=%s d_model=d_ff=%s\n' \
    "$model_name" "$SEQ_LEN" "$pred_len" "$width"
  "$PYTHON" -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --data_path ETTh1.csv \
    --model_id "ETTh1_dual_${SEQ_LEN}_${pred_len}_w${width}" \
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
    --decomp_moving_avg "$MOVING_AVG" \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --use_norm 1 \
    --des DualDecomp \
    --itr 1
done
