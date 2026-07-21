export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_three
three_patch_len=16
three_stride=8
three_patch_layers=2
three_fusion_hidden=256
three_head_dropout=0.1
three_gamma_init=0.1
three_patch_loss_weight=0.2
three_joint_loss_weight=0.2
three_base_loss_weight=0.1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_three_96_96 \
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
  --three_patch_len $three_patch_len \
  --three_stride $three_stride \
  --three_patch_layers $three_patch_layers \
  --three_fusion_hidden $three_fusion_hidden \
  --three_head_dropout $three_head_dropout \
  --three_gamma_init $three_gamma_init \
  --three_patch_loss_weight $three_patch_loss_weight \
  --three_joint_loss_weight $three_joint_loss_weight \
  --three_base_loss_weight $three_base_loss_weight \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_three_96_192 \
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
  --three_patch_len $three_patch_len \
  --three_stride $three_stride \
  --three_patch_layers $three_patch_layers \
  --three_fusion_hidden $three_fusion_hidden \
  --three_head_dropout $three_head_dropout \
  --three_gamma_init $three_gamma_init \
  --three_patch_loss_weight $three_patch_loss_weight \
  --three_joint_loss_weight $three_joint_loss_weight \
  --three_base_loss_weight $three_base_loss_weight \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_three_96_336 \
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
  --three_patch_len $three_patch_len \
  --three_stride $three_stride \
  --three_patch_layers $three_patch_layers \
  --three_fusion_hidden $three_fusion_hidden \
  --three_head_dropout $three_head_dropout \
  --three_gamma_init $three_gamma_init \
  --three_patch_loss_weight $three_patch_loss_weight \
  --three_joint_loss_weight $three_joint_loss_weight \
  --three_base_loss_weight $three_base_loss_weight \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_three_96_720 \
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
  --three_patch_len $three_patch_len \
  --three_stride $three_stride \
  --three_patch_layers $three_patch_layers \
  --three_fusion_hidden $three_fusion_hidden \
  --three_head_dropout $three_head_dropout \
  --three_gamma_init $three_gamma_init \
  --three_patch_loss_weight $three_patch_loss_weight \
  --three_joint_loss_weight $three_joint_loss_weight \
  --three_base_loss_weight $three_base_loss_weight \
  --itr 1
