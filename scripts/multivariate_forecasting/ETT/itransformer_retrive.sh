#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=10
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Historical pairs come only from the training split. Training sample indices
# enforce memory_start + seq_len + pred_len <= current_start + seq_len.
# PRED_LENS="96 192 336 720" runs all horizons. Defaults to one experiment.
# USE_RETRIEVAL=0: baseline; RETRIEVAL_USE_FUTURE=0: past-only values;
# RETRIEVAL_USE_GATE=0: additive fusion; RETRIEVAL_WEIGHTED=0: uniform Top-K.
PYTHON="${PYTHON:-python}"
read -r -a horizons <<< "${PRED_LENS:-96}"
for pred_len in "${horizons[@]}"; do
  "$PYTHON" -u run.py \
    --is_training 1 \
    --model_id "ETTh1_VR_${SEQ_LEN:-96}_${pred_len}" \
    --model itransformer_retrieval \
    --data ETTh1 \
    --root_path "${ROOT_PATH:-./dataset/ETT-small/}" \
    --data_path ETTh1.csv \
    --features M \
    --seq_len "${SEQ_LEN:-96}" \
    --label_len 48 \
    --pred_len "$pred_len" \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --n_heads 8 --d_model 256 --d_ff 256 \
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
    --dropout 0.1 --batch_size 32 --train_epochs 10 --patience 3 \
    --learning_rate 0.0001 --lradj type1 --use_norm 1 \
    --des VariableWiseRetrieval --itr 1
done
