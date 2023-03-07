export MODEL_NAME="runwayml/stable-diffusion-v1-5"
accelerate launch --gpu_ids 0,1 src/finetune_lora_baseline_0_14ver.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/cat_single/3  \
          --class_data_dir=./gen_reg/samples_cat/samples/ \
          --output_dir=./logs/cat_lora_textural_noreg_control  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="a cat" \
          --resolution=512  \
          --train_batch_size=1  \
          --learning_rate=1e-4  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --instance_prompt "a <new1> cat is standing on the step"
          