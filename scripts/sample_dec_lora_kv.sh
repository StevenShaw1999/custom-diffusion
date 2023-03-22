CUDA_VISIBLE_DEVICES=1 python src/sample_dec_lora_kv.py \
                            --delta_ckpt /data1/jiayu_xiao/project/custom-diffusion/logs/man_personal_textural_dec_plus/delta.bin \
                            --ckpt /data1/jiayu_xiao/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819 \
                            --prompt "a <new1> man playing guitar, best quality, highly detailed"