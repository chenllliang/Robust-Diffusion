export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#--max_train_steps=15000 \
#  --use_ema \
# --text_embed_dir ./text_embed.bin \
#  --text_embed_dir="./text_embed_linear_p_beta1_n5.bin" \
#  --text_augment="crop_swap" \

accelerate launch --mixed_precision="fp16"  fine_tune_text2img.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="./coco_mini/train" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --num_train_epochs=2 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --seed 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="coco-model" 
