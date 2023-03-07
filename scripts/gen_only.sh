#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption
# 2nd arg: path to target images
# 3rd arg: path where generated images are saved
# 4rth arg: name of the experiment
# 5th arg: config name
# 6th arg: pretrained model path

ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done


python -u sample.py \
        --n_samples 5 \
        --n_iter 40 \
        --scale 6 \
        --ddim_steps 50  \
        --ckpt ${ARRAY[5]} \
        --ddim_eta 1. \
        --outdir "${ARRAY[2]}" \
        --prompt "photo of a ${ARRAY[0]}" 


