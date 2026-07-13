#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_decom
seq_len=96

# Single-scale DFT decomposition on the original input history; no input pyramid.
decomp_dft_top_k=5
decomp_hidden=128
decomp_tcn_layers=2
decomp_fluctuation_gate_bias=-2.0

# ETTh1 contains seven variables, so at most six external variables exist for
# each target. The second stage selects component-lag sources from this list.
decomp_lags=0,1,2,4,8
decomp_variate_top_k=6
decomp_top_k=3
decomp_router_temperature=1.0
decomp_cross_gate_bias=-2.5

# Auxiliary objectives: trend-dominant component supervision, self forecast,
# marginal utility, negative-transfer safety and routing sparsity.
decomp_trend_loss=0.2
decomp_fluctuation_loss=0.05
decomp_self_loss=0.1
decomp_utility_loss=0.05
decomp_safe_loss=0.05
decomp_entropy_loss=0.001

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_decom_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --e_layers 2 \
    --d_model $decomp_hidden \
    --d_ff 256 \
    --decomp_dft_top_k $decomp_dft_top_k \
    --decomp_lags $decomp_lags \
    --decomp_hidden $decomp_hidden \
    --decomp_tcn_layers $decomp_tcn_layers \
    --decomp_fluctuation_gate_bias $decomp_fluctuation_gate_bias \
    --decomp_variate_top_k $decomp_variate_top_k \
    --decomp_top_k $decomp_top_k \
    --decomp_router_temperature $decomp_router_temperature \
    --decomp_cross_gate_bias $decomp_cross_gate_bias \
    --decomp_trend_loss $decomp_trend_loss \
    --decomp_fluctuation_loss $decomp_fluctuation_loss \
    --decomp_self_loss $decomp_self_loss \
    --decomp_utility_loss $decomp_utility_loss \
    --decomp_safe_loss $decomp_safe_loss \
    --decomp_entropy_loss $decomp_entropy_loss \
    --des 'CURF' \
    --itr 1
done
