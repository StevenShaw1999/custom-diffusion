# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from src import diffuser_training 
from diffusers import UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
import cv2
from PIL import Image

#from lora_diffusion import monkeypatch_or_replace_lora, tune_lora_scale


def sample(ckpt, delta_ckpt, from_file, prompt, compress, freeze_model):
    apply_hed = HEDdetector()
    ori_image = load_image(
    "/data1/jiayu_xiao/project/custom-diffusion/data/fame/yann-lecun.jpg"
    )
    model_id = ckpt
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, controlnet=controlnet
    )
    #pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    image = np.array(ori_image)
    input_image = HWC3(image)
    detected_map = apply_hed.forward(resize_image(input_image, 256))
    detected_map = HWC3(detected_map)
    img = resize_image(detected_map, 512)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR) #/ 255
    detected_map = Image.fromarray(detected_map)

    outdir = 'outputs/txt2img-samples'
    os.makedirs(outdir, exist_ok=True)
    print(compress)
    print(freeze_model)
    if delta_ckpt is not None:
        diffuser_training.load_model_new(pipe.text_encoder, pipe.tokenizer, delta_ckpt)
        outdir = os.path.dirname(delta_ckpt)
    pipe.unet.load_attn_procs(outdir)
    pipe.to('cuda')
    if prompt is not None:
        image_list = []
        for i in range(1, 4):
            generator = [torch.Generator(device="cpu").manual_seed(j * i) for j in [5,6,7]]
            images = pipe(prompt=[prompt for i in range(3)], image=detected_map, num_inference_steps=20, guidance_scale=6., \
                          eta=1., controlnet_conditioning_scale=0.0, generator=generator).images
            #images = pipe([prompt]*5, num_inference_steps=200, guidance_scale=6., eta=1.).images
            images = np.hstack([np.array(x) for x in images])
            image_list.append(images)
        
        image_list = np.concatenate([np.array(x) for x in image_list])
        #print(image_list.shape)

        plt.imshow(image_list)
        plt.axis("off")
        plt.savefig(f'{outdir}/{prompt}.png', bbox_inches='tight')
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [5 * [prompt] for prompt in data]

        for prompt in data:
            images = pipe(prompt, num_inference_steps=200, guidance_scale=6., eta=1.).images
            images = np.hstack([np.array(x) for x in images], 0)
            plt.imshow(images)
            plt.axis("off")
            plt.savefig(f'{outdir}/{prompt[0]}.png', bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--from-file', help='path to prompt file', default='./',
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.freeze_model)
