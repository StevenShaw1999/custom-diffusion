export MODEL_NAME="/data1/jiayu_xiao/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819"
accelerate launch --gpu_ids 3,4 --main_process_port 9998 src/finetune_personalize.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/fame  \
          --class_data_dir=./gen_reg/samples_man/samples/ \
          --output_dir=./logs/man_lora_textural_noreg_8_control_180_grad_1.0_mask_base1  \
          --with_prior_preservation --prior_loss_weight=0.0 \
          --class_prompt="a man" \
          --resolution=512  \
          --train_batch_size=1  \
          --learning_rate=5e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>+<new2>+<new3>+<new4>+<new5>" \
          --instance_prompt "a man is smiling for the camera"
          