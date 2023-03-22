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

#from lora_diffusion import monkeypatch_or_replace_lora, tune_lora_scale


def create_custom_diffusion_decouple(unet, freeze_model):
    """for name, params in unet.named_parameters():
        if freeze_model == 'crossattn':
            if 'attn2' in name:
                params.requires_grad = True
                print(name)
            else:
                params.requires_grad = False
        else:
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params.requires_grad = True
                print(name)
            else:
                params.requires_grad = False"""

    def new_forward(self, hidden_states, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        context = None
        
        if 'wher2swap' in kwargs:
            wher2swap = kwargs['wher2swap']
        if 'decouple_wher' in kwargs:
            decouple_wher = kwargs['decouple_wher']


        if 'context' in kwargs:
            context = kwargs['context']
        elif 'encoder_hidden_states' in kwargs:
            context = kwargs['encoder_hidden_states']
        if context is not None:
            crossattn = True

        #print(crossattn)

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        if crossattn:
            modifier = torch.ones_like(key)
            modifier[:, :1, :] = modifier[:, :1, :]*0.
            key = modifier*key + (1-modifier)*key.detach()
            value = modifier*value + (1-modifier)*value.detach()

            key_unc, key = torch.chunk(key, 2, dim=0)
            value_unc, value = torch.chunk(value, 2, dim=0)

            key, key_dec = key[:-2], key[-2:]
            value, value_dec = value[:-2], value[-2:]


            value_dec_cpy = value_dec[:, 1:decouple_wher+1, :]
            #value_dec = value_dec[:, 1:2, :]
            sub_value_dec = value_dec_cpy.permute(2, 0, 1)
            #print(value.shape, sub_value_dec.shape)
            attn_val = torch.einsum('aij, jkb -> aikb', value, sub_value_dec)
            #print(attn_val.shape)
            attn_val_probs = attn_val.softmax(dim=-1)
            attn_val_probs_mask = torch.zeros_like(attn_val_probs)
            val_mask = torch.zeros_like(value)
            for num, item in enumerate(wher2swap):
                attn_val_probs_mask[num, item, 0, : ] = 1
                attn_val_probs_mask[num, item+1, 1, :] = 1
                val_mask[num, item, :] = 1
                val_mask[num, item+1, :] = 1
            
            #print(value.shape, attention_probs_mask.shape)
            sub_val_dec = torch.einsum('aijk, jkb -> aib', attn_val_probs * attn_val_probs_mask, value_dec_cpy)
            
            value = val_mask * sub_val_dec + (1 - val_mask) * value
            
            value = torch.cat([value_unc[:-2], value])
            key = torch.cat([key_unc[:-2], key])

            #print(key.shape, query.shape, value.shape)

        dim = query.shape[-1]

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        
        #query, query_decouple = None

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        attention_probs = self.get_attention_scores(query, key)
        """if crossattn:
            print(attention_probs.shape, value.shape)
            print(attention_probs[0, 2048, :])
            #print(attention_probs[0, :10, 5])
            exit()"""
        """if crossattn:
            print('Attn probs', attention_probs.shape)
            print('Value', value.shape)"""

        hidden_states = torch.bmm(attention_probs, value)

        #if crossattn:
        #    print('hidden states', hidden_states.shape)

        hidden_states = self.batch_to_head_dim(hidden_states)
        
        #if crossattn:
        #    print('hidden states 2', hidden_states.shape)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        #if crossattn:
        #    print('hidden states 3', hidden_states.shape)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        #if crossattn:
        #    print('hidden states 4', hidden_states.shape)
        #    exit()
        #print(hidden_states.shape)
        #exit()
        return hidden_states

    def change_forward(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer)
    change_forward(unet)
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
    
    pipe.unet = create_custom_diffusion_decouple(pipe.unet, freeze_model)
    #pipe.unet.load_attn_procs(outdir)
    #exit()
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
                          eta=1., generator=generator, cross_attention_kwargs={'wher2swap': [2,2,2], 'decouple_wher': 3}).images
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
