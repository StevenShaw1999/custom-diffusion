import torch
import torch.nn as nn
from diffusers.models.attention import CrossAttention
class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)


class LoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, rank=8):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        #self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        #if cross_attention_dim is not None:
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        #self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: CrossAttention, hidden_states, **kwargs
    ):

        batch_size, sequence_length, _ = hidden_states.shape
        crossattn = False
        context = None
        
        if 'inference' in kwargs:
            is_inf = kwargs['inference']
        else:
            is_inf = 0

        if 'wher2swap' in kwargs:
            wher2swap = kwargs['wher2swap']
        if 'decouple_wher' in kwargs:
            decouple_wher = kwargs['decouple_wher']

        if 'scale' in kwargs:
            scale = kwargs['scale']
        else: scale = 1.0

        if 'context' in kwargs:
            context = kwargs['context']
        elif 'encoder_hidden_states' in kwargs:
            context = kwargs['encoder_hidden_states']
        if context is not None:
            crossattn = True

        #attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states) #+ scale * self.to_q_lora(hidden_states)
        context = context if context is not None else hidden_states

        if crossattn:
            #print(scale)
            key = attn.to_k(context) + scale * self.to_k_lora(context)
            value = attn.to_v(context) + scale * self.to_v_lora(context)
        
        else:
            key = attn.to_k(context)
            value = attn.to_v(context)
        

        if crossattn:
            #print('Here')
            modifier = torch.ones_like(key)
            
            # print(key.shape)
            modifier[:, :1, :] = modifier[:, :1, :]*0.
            key = modifier*key + (1-modifier)*key.detach()
            value = modifier*value + (1-modifier)*value.detach()

            if is_inf:
                key_unc, key = torch.chunk(key, 2, dim=0)
                value_unc, value = torch.chunk(value, 2, dim=0)

            key, key_dec = key[:-2], key[-2:]
            value, value_dec = value[:-2], value[-2:]

            value_dec = value_dec[:, 1:decouple_wher+1, :]
            #value_dec = value_dec[:, 1:2, :]
            sub_value_dec = value_dec.permute(2, 0, 1)
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
            sub_val_dec = torch.einsum('aijk, jkb -> aib', attn_val_probs * attn_val_probs_mask, value_dec)
            
            value = val_mask * sub_val_dec + (1 - val_mask) * value

            if is_inf:
                value = torch.cat([value_unc[:-2], value])
                key = torch.cat([key_unc[:-2], key])


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) #+ scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states