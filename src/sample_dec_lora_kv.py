# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from src import diffuser_training 
from diffusers.models.attention import CrossAttention
from diffusers import UniPCMultistepScheduler, DPMSolverMultistepScheduler
from lora_custom_kv import LoRACrossAttnProcessor
from diffusers.loaders import AttnProcsLayers
from collections import defaultdict

#from lora_diffusion import monkeypatch_or_replace_lora, tune_lora_scale


def create_custom_diffusion_lora_decouple(unet, model_file):

    state_dict = torch.load(model_file, map_location="cpu")

    attn_processors = {}

    is_lora = all("lora" in k for k in state_dict.keys())

    if is_lora:
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value

        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict["to_k_lora.down.weight"].shape[0]
            cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
            hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

            attn_processors[key] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
            )
            attn_processors[key].load_state_dict(value_dict)

    else:
        raise ValueError(f"{model_file} does not seem to be in the correct format expected by LoRA training.")

    # set correct dtype & device
    attn_processors = {k: v.to(device=unet.device, dtype=unet.dtype) for k, v in attn_processors.items()}

    # set layers
    unet.set_attn_processor(attn_processors)

    #change_forward(unet)
    return unet


def sample(ckpt, delta_ckpt, from_file, prompt, compress, freeze_model):
    model_id = ckpt
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    outdir = 'outputs/txt2img-samples'
    os.makedirs(outdir, exist_ok=True)
    print(compress)
    print(freeze_model)
    if delta_ckpt is not None:
        diffuser_training.load_model_new(pipe.text_encoder, pipe.tokenizer, delta_ckpt)
        outdir = os.path.dirname(delta_ckpt)
    
    
    pipe.unet = create_custom_diffusion_lora_decouple(pipe.unet, '/data1/jiayu_xiao/project/custom-diffusion/logs/man_personal_textural_dec_lora/pytorch_lora_weights.bin')
    
    #pipe.unet.load_attn_procs('/data1/jiayu_xiao/project/custom-diffusion/logs/man_personal_textural_dec_lora')

    if prompt is not None:
        image_list = []
        for i in range(1, 4):
            generator = [torch.Generator(device="cpu").manual_seed(j * i) for j in [5,6,7]]
            prompts_new = [prompt]*3
            #
            prompts_new.append('<new2> <new3> <new4>')
            prompts_new.append('<new5> <new6> <new7>')
            images = pipe.inference(prompts_new, num_inference_steps=20, guidance_scale=6., 
                          negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 5,
                          eta=1., generator=generator, cross_attention_kwargs={'wher2swap': [2,2,2], 'decouple_wher': 3, 'scale': 1.0, 'inference': 1}).images
            #  
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
