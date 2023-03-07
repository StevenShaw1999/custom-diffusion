export MODEL_NAME="CompVis/stable-diffusion-v1-4"


accelerate launch src/diffuser_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/dog_single/1  \
          --class_data_dir=./gen_reg/samples_dog/samples/ \
          --output_dir=./logs/dog_150_epoch_1  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="dog" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=150 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --gradient_accumulation_steps=2 --gradient_checkpointing \
          --instance_prompt "a photo of a <new1> dog"

accelerate launch src/diffuser_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/dog_single/2  \
          --class_data_dir=./gen_reg/samples_dog/samples/ \
          --output_dir=./logs/dog_150_epoch_2  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="dog" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=150 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --gradient_accumulation_steps=2 --gradient_checkpointing \
          --instance_prompt "a photo of a <new1> dog"



accelerate launch src/diffuser_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/dog_single/3  \
          --class_data_dir=./gen_reg/samples_dog/samples/ \
          --output_dir=./logs/dog_150_epoch_3  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="dog" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=150 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --gradient_accumulation_steps=2 --gradient_checkpointing \
          --instance_prompt "a photo of a <new1> dog"


accelerate launch src/diffuser_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/dog_single/4  \
          --class_data_dir=./gen_reg/samples_dog/samples/ \
          --output_dir=./logs/dog_150_epoch_4  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --class_prompt="dog" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=150 \
          --num_class_images=200 \
          --scale_lr \
          --modifier_token "<new1>" \
          --gradient_accumulation_steps=2 --gradient_checkpointing \
          --instance_prompt "a photo of a <new1> dog"

