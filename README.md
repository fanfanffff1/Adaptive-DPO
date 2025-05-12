# Adaptive-DPO
Adaptive-DPO
'''
accelerate launch dpo_train_un.py \
  --pretrained_model_name_or_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuzhenyang-240108540154/zhanglingfan/stable-diffusion-v1-5 \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=64 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --rank=8 \
  --learning_rate=1e-5 \
  --lr_warmup_steps=0 \
  --num_train_epochs=1 \
  --checkpointing_steps=3000 \
  --seed="0" \
  --dataset_name=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuzhenyang-240108540154/zhanglingfan/datasets/preprocess_data \
  --cache_dir=/dockerdata \
  --dataset_split_name="train" \
  --beta_dpo=1000 \
  --output_dir=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuzhenyang-240108540154/zhanglingfan/rebuttal/clean_save_std \
  --use_self_logit_reweight=True \
  --rewight_intercept=1 \
  --rewight_temparature=10 \
  --use_ema=True \
  --logits_rho=15 \
  --use_logits_normalization=True \
  --use_margin=True \
  --margin_type=quadratic
  '''
