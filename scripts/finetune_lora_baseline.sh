export MODEL_NAME="runwayml/stable-diffusion-v1-5"
accelerate launch --gpu_ids 0,1 src/finetune_lora_baseline_0_14ver.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/fame  \
          --class_data_dir=./gen_reg/samples_man/samples/ \
          --output_dir=./logs/man_lora_textural_noreg_control_new  \
          --with_prior_preservation --prior_loss_weight=0.0 \
          --class_prompt="a man" \
          --resolution=512  \
          --train_batch_size=1  \
          --learning_rate=1e-4  \
          --lr_warmup_steps=0 \
          --max_train_steps=400 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --instance_prompt "a <new1> man is smiling for the camera"
          