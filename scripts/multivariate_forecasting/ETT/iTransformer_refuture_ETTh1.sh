export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer_refuture
refuture_splits=0.25,0.5,0.75
refuture_steps=2
refuture_step_size=0.2
refuture_anchor_weight=0.1
refuture_max_update=0.5
refuture_learnable_step=1
refuture_differentiable=0
refuture_base_loss_weight=0.2
refuture_consistency_loss_weight=0.05
refuture_safe_loss_weight=0.1
refuture_update_loss_weight=0.01

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_refuture_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --refuture_splits $refuture_splits \
  --refuture_steps $refuture_steps \
  --refuture_step_size $refuture_step_size \
  --refuture_anchor_weight $refuture_anchor_weight \
  --refuture_max_update $refuture_max_update \
  --refuture_learnable_step $refuture_learnable_step \
  --refuture_differentiable $refuture_differentiable \
  --refuture_base_loss_weight $refuture_base_loss_weight \
  --refuture_consistency_loss_weight $refuture_consistency_loss_weight \
  --refuture_safe_loss_weight $refuture_safe_loss_weight \
  --refuture_update_loss_weight $refuture_update_loss_weight \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_refuture_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --refuture_splits $refuture_splits \
  --refuture_steps $refuture_steps \
  --refuture_step_size $refuture_step_size \
  --refuture_anchor_weight $refuture_anchor_weight \
  --refuture_max_update $refuture_max_update \
  --refuture_learnable_step $refuture_learnable_step \
  --refuture_differentiable $refuture_differentiable \
  --refuture_base_loss_weight $refuture_base_loss_weight \
  --refuture_consistency_loss_weight $refuture_consistency_loss_weight \
  --refuture_safe_loss_weight $refuture_safe_loss_weight \
  --refuture_update_loss_weight $refuture_update_loss_weight \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_refuture_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --refuture_splits $refuture_splits \
  --refuture_steps $refuture_steps \
  --refuture_step_size $refuture_step_size \
  --refuture_anchor_weight $refuture_anchor_weight \
  --refuture_max_update $refuture_max_update \
  --refuture_learnable_step $refuture_learnable_step \
  --refuture_differentiable $refuture_differentiable \
  --refuture_base_loss_weight $refuture_base_loss_weight \
  --refuture_consistency_loss_weight $refuture_consistency_loss_weight \
  --refuture_safe_loss_weight $refuture_safe_loss_weight \
  --refuture_update_loss_weight $refuture_update_loss_weight \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_refuture_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --refuture_splits $refuture_splits \
  --refuture_steps $refuture_steps \
  --refuture_step_size $refuture_step_size \
  --refuture_anchor_weight $refuture_anchor_weight \
  --refuture_max_update $refuture_max_update \
  --refuture_learnable_step $refuture_learnable_step \
  --refuture_differentiable $refuture_differentiable \
  --refuture_base_loss_weight $refuture_base_loss_weight \
  --refuture_consistency_loss_weight $refuture_consistency_loss_weight \
  --refuture_safe_loss_weight $refuture_safe_loss_weight \
  --refuture_update_loss_weight $refuture_update_loss_weight \
  --itr 1
