CUDA_VISIBLE_DEVICES=5 python src/sample_diffuser_with_lora.py \
                            --delta_ckpt /data1/jiayu_xiao/project/custom-diffusion/logs/man_lora_textural_noreg_8_control_180_grad_1.0_mask_base1/delta.bin \
                            --ckpt /data1/jiayu_xiao/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819 \
                            --prompt "a man is smiling for the camera, best quality, extremely detailed"