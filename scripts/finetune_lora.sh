export MODEL_NAME="runwayml/stable-diffusion-v1-5"
accelerate launch --gpu_ids 5 src/finetune_lora.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/fame  \
          --class_data_dir=./gen_reg/samples_man/samples/ \
          --output_dir=./logs/man_lora_textural_reg  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="a man is smiling for the camera" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=300 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --gradient_accumulation_steps=2 --gradient_checkpointing \
          --instance_prompt "a <new1> man is smiling for the camera"