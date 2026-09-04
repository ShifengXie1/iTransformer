export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_multihead

# Horizon-conditioned explicit relation heads. All seven ETTh1 variables are
# tokens in one iTransformer; future steps dynamically route across four heads.
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --mh_relation_heads 4 \
  --mh_head_dim 0 \
  --mh_gate_dim 32 \
  --mh_horizon_temperature 1.0 \
  --mh_horizon_prior_strength 1.0 \
  --mh_residual_init 0.1 \
  --mh_exclude_self 1 \
  --mh_diversity_loss_weight 0.01 \
  --mh_balance_loss_weight 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --mh_relation_heads 4 \
  --mh_head_dim 0 \
  --mh_gate_dim 32 \
  --mh_horizon_temperature 1.0 \
  --mh_horizon_prior_strength 1.0 \
  --mh_residual_init 0.1 \
  --mh_exclude_self 1 \
  --mh_diversity_loss_weight 0.01 \
  --mh_balance_loss_weight 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --mh_relation_heads 4 \
  --mh_head_dim 0 \
  --mh_gate_dim 32 \
  --mh_horizon_temperature 1.0 \
  --mh_horizon_prior_strength 1.0 \
  --mh_residual_init 0.1 \
  --mh_exclude_self 1 \
  --mh_diversity_loss_weight 0.01 \
  --mh_balance_loss_weight 0.001 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --mh_relation_heads 4 \
  --mh_head_dim 0 \
  --mh_gate_dim 32 \
  --mh_horizon_temperature 1.0 \
  --mh_horizon_prior_strength 1.0 \
  --mh_residual_init 0.1 \
  --mh_exclude_self 1 \
  --mh_diversity_loss_weight 0.01 \
  --mh_balance_loss_weight 0.001 \
  --itr 1
