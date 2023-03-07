export MODEL_NAME="CompVis/stable-diffusion-v1-4"
accelerate launch src/diffuser_training_new.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/cat_single/3  \
          --class_data_dir=./gen_reg/samples_cat/samples/ \
          --output_dir=./logs/cat_3_masked_prior  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="cat" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=160 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --gradient_accumulation_steps=2 --gradient_checkpointing \
          --instance_prompt "a <new1> standing on a step"