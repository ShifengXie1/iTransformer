#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=10

# Run from any directory; run.py already records training output under logs/.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

PYTHON="${PYTHON:-python}"
ROOT_PATH="${ROOT_PATH:-./dataset/ETT-small/}"
SEQ_LEN="${SEQ_LEN:-96}"
LAMBDA_JOINT="${LAMBDA_JOINT:-0.1}"
ALIGNMENT_MODE="${ALIGNMENT_MODE:-joint}"
TEMPORAL_KEEP_RATIO="${TEMPORAL_KEEP_RATIO:-0.5}"
VARIATE_KEEP_RATIO="${VARIATE_KEEP_RATIO:-0.5}"
JOINT_WEIGHTING="${JOINT_WEIGHTING:-sqrt_eigen_product}"
CORRELATION_EPS="${CORRELATION_EPS:-1e-6}"
CORR_STANDARDIZE_LABELS="${CORR_STANDARDIZE_LABELS:-1}"

# Run correlation alignment at all four horizons with the original backbone.
# PRED_LENS="336" selects a single 336-step run.
# ALIGNMENT_MODE=none or LAMBDA_JOINT=0 disables the auxiliary objective.
width=256
model_name=itransformer_correlation
read -r -a horizons <<< "${PRED_LENS:-96 192 336 720}"
for pred_len in "${horizons[@]}"; do
  case "$pred_len" in
    96|192|336|720) ;;
    *) echo "Unsupported PRED_LENS entry: $pred_len" >&2; exit 1 ;;
  esac
  printf '\nRunning model=%s seq_len=%s pred_len=%s d_model=d_ff=%s alignment=%s lambda_joint=%s\n' \
    "$model_name" "$SEQ_LEN" "$pred_len" "$width" "$ALIGNMENT_MODE" "$LAMBDA_JOINT"
  "$PYTHON" -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --data_path ETTh1.csv \
    --model_id "ETTh1_corr_${SEQ_LEN}_${pred_len}_w${width}" \
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
    --lambda_joint "$LAMBDA_JOINT" \
    --alignment_mode "$ALIGNMENT_MODE" \
    --temporal_keep_ratio "$TEMPORAL_KEEP_RATIO" \
    --variate_keep_ratio "$VARIATE_KEEP_RATIO" \
    --joint_weighting "$JOINT_WEIGHTING" \
    --correlation_eps "$CORRELATION_EPS" \
    --corr_standardize_labels "$CORR_STANDARDIZE_LABELS" \
    --dropout 0.1 \
    --batch_size 32 \
    --train_epochs 10 \
    --patience 3 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --use_norm 1 \
    --des JointCorrelation \
    --itr 1
done
