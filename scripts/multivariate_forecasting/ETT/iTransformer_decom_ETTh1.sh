#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; run.py already records training output under logs/.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
SEQ_LEN="${SEQ_LEN:-96}"
MOVING_AVG="${MOVING_AVG:-25}"

# Same per-branch widths as the original ETTh1 script, hence about twice its
# parameter count. This first run tests the architecture, not parameter parity.
# PRED_LENS="96" can be used for a single-horizon pilot.
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192) width=256 ;;
    336|720) width=512 ;;
    *) echo "Unsupported PRED_LENS entry: $pred_len" >&2; exit 1 ;;
  esac
  "$PYTHON" -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --data_path ETTh1.csv \
    --model_id "ETTh1_dual_${SEQ_LEN}_${pred_len}" \
    --model iTransformer_decom \
    --data ETTh1 \
    --features M \
    --seq_len "$SEQ_LEN" \
    --pred_len "$pred_len" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --n_heads 8 \
    --d_model "$width" \
    --d_ff "$width" \
    --decomp_moving_avg "$MOVING_AVG" \
    --use_norm 1 \
    --des DualDecomp \
    --itr 1
done
