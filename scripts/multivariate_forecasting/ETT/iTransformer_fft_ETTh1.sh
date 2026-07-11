export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer_fft
base_patch_len=48
period_query_num=4
num_channel_tokens=4
Seq_len=336

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_fft_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $Seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --base_patch_len $base_patch_len \
  --period_query_num $period_query_num \
  --num_channel_tokens $num_channel_tokens \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_fft_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $Seq_len \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --base_patch_len $base_patch_len \
  --period_query_num $period_query_num \
  --num_channel_tokens $num_channel_tokens \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_fft_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $Seq_len \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --base_patch_len $base_patch_len \
  --period_query_num $period_query_num \
  --num_channel_tokens $num_channel_tokens \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_fft_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $Seq_len \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --base_patch_len $base_patch_len \
  --period_query_num $period_query_num \
  --num_channel_tokens $num_channel_tokens \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
