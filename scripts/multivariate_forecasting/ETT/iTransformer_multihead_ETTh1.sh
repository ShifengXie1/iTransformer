export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_multihead

# Four views = original anchor + high-frequency + mid-frequency + trend.
# A zero residual initialization makes epoch zero exactly match iTransformer.
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
  --num_token_heads 4 \
  --token_scales auto \
  --use_dynamic_mask 1 \
  --token_mask_hidden 64 \
  --token_temperature 1.0 \
  --view_attention_heads 4 \
  --view_residual_init 0.0 \
  --gate_temperature 1.0 \
  --fusion_type dynamic \
  --lambda_redundancy 0 \
  --lambda_mask_diversity 0 \
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
  --num_token_heads 4 \
  --token_scales auto \
  --use_dynamic_mask 1 \
  --token_mask_hidden 64 \
  --token_temperature 1.0 \
  --view_attention_heads 4 \
  --view_residual_init 0.0 \
  --gate_temperature 1.0 \
  --fusion_type dynamic \
  --lambda_redundancy 0 \
  --lambda_mask_diversity 0 \
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
  --num_token_heads 4 \
  --token_scales auto \
  --use_dynamic_mask 1 \
  --token_mask_hidden 64 \
  --token_temperature 1.0 \
  --view_attention_heads 4 \
  --view_residual_init 0.0 \
  --gate_temperature 1.0 \
  --fusion_type dynamic \
  --lambda_redundancy 0 \
  --lambda_mask_diversity 0 \
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
  --num_token_heads 4 \
  --token_scales auto \
  --use_dynamic_mask 1 \
  --token_mask_hidden 64 \
  --token_temperature 1.0 \
  --view_attention_heads 4 \
  --view_residual_init 0.0 \
  --gate_temperature 1.0 \
  --fusion_type dynamic \
  --lambda_redundancy 0 \
  --lambda_mask_diversity 0 \
  --itr 1
