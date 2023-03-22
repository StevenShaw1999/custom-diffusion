export MODEL_NAME="/data1/jiayu_xiao/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819"
accelerate launch --gpu_ids 1,3 --main_process_port 9998 src/finetune_dec_plus.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/fame  \
          --output_dir=./logs/man_personal_textural_dec_plus  \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=5e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>+<new2>+<new3>+<new4>+<new5>+<new6>+<new7>" \
          --instance_prompt "a <new1> man is smiling for the camera"
          