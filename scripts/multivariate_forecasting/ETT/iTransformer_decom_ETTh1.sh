#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_decom
seq_len=96

# Original TimeMixer ETTh1 self predictor: moving average and 96/48/24/12 scales.
decomp_moving_avg=25
decomp_hidden=16
decomp_d_ff=32
decomp_mixing_layers=2
decomp_down_sampling_layers=3
decomp_down_sampling_window=2
decomp_down_sampling_method=avg

# ETTh1 contains seven variables, so at most six external variables exist for
# each target. Stage two selects component-scale-lag sources from this list.
decomp_lags=0,1,2,4,8
decomp_variate_top_k=6
decomp_top_k=3
decomp_router_temperature=1.0
decomp_cross_gate_bias=-2.5

# Auxiliary objectives preserve the TimeMixer self baseline and train only
# useful, non-destructive cross-variate residual corrections.
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
    --d_ff $decomp_d_ff \
    --decomp_moving_avg $decomp_moving_avg \
    --decomp_lags $decomp_lags \
    --decomp_hidden $decomp_hidden \
    --decomp_d_ff $decomp_d_ff \
    --decomp_mixing_layers $decomp_mixing_layers \
    --decomp_down_sampling_layers $decomp_down_sampling_layers \
    --decomp_down_sampling_window $decomp_down_sampling_window \
    --decomp_down_sampling_method $decomp_down_sampling_method \
    --decomp_variate_top_k $decomp_variate_top_k \
    --decomp_top_k $decomp_top_k \
    --decomp_router_temperature $decomp_router_temperature \
    --decomp_cross_gate_bias $decomp_cross_gate_bias \
    --decomp_self_loss $decomp_self_loss \
    --decomp_utility_loss $decomp_utility_loss \
    --decomp_safe_loss $decomp_safe_loss \
    --decomp_entropy_loss $decomp_entropy_loss \
    --des 'TMCR' \
    --itr 1
done
