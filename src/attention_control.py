import torch
import abc

LOW_RESOURCE = False 

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.flag:
            if self.cur_att_layer >= self.num_uncond_att_layers:
                if LOW_RESOURCE:
                    attn = self.forward(attn, is_cross, place_in_unet)
                else:
                    h = attn.shape[0]
                    # shape like [16, 4096, 4096]: self attention, num of (2 x heads) x (hxw) x (hxw)
                    # shape like [16, 4096, 77]: cross attention, num of (2 x heads) x (hxw) x num of tokens, 77 is the num of tokens of the padded text

                    # why extract the second half of attention 
                    # reason is that the textual input is in form of 
                    # [uncond_embed, cond_embed], the first half do not 
                    # contribute to the attention visualization
                    attn = self.forward(attn, is_cross, place_in_unet)
            self.cur_att_layer += 1
            #print(self.cur_att_layer)
            # collect attention step util the total cross attn layer count 
            #print(self.num_att_layers, self.num_uncond_att_layers, self.cur_att_layer)
            if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
                #print('Full')
                self.cur_att_layer = 0
                self.cur_step += 1
                # call between_steps() function
                self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, prompt_mask=None):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.flag = False
        self.prompt_mask = prompt_mask

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # filter of resolution
        if attn.shape[1] == 16 ** 2 and is_cross:  # avoid memory overhead
            self.step_store[key].append(attn.data)
        return attn

    def between_steps(self):
        # AttentionStore is inherited from AttentionControl class 
        #if len(self.attention_store) == 0:
        self.attention_store = self.step_store
        """else:
            # add temp attn_scores to the stored attn dict
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    print(len())
                    self.attention_store[key][i] = self.step_store[key][i]"""
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # mean score according to the current time step
        average_attention = {key: [item  for item in self.attention_store[key]] for key in self.attention_store}
        self.cur_step = 0
        #print('!!!!!!!')
        #print(self.num_att_layers, self.num_uncond_att_layers, self.cur_att_layer)
        #print(self.cur_step)
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

def register_attention_control(unet, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            #print(context.size())

            # image embed: [2, 64x64, 320] 2 is the dual way of condi/uncondi path
            # text embed: [2, 77, 768]
            # both are embedded to dimention 40
            # num of multi-head attention is 8
            # k: [8*2, 77, 40] for text embed
            k = self.to_k(context)
            v = self.to_v(context)
            #if is_cross:
            #    print(q.size(), k.size(), v.size())
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)
            #if is_cross:
            #    print(q.size(), k.size(), v.size())
            #    exit()
            #print(q.size(), k.size())
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            # The controller is inherited from AttentionControl Class
            # if the controller is defined as AttentionStore
            # recall the forward function of AttentionStore class
            # store the attention weight of different layers in AttentionStore.step_store dict

            # input of form (attn, is_cross: bool, place_in_unet: str) 
            # call the __call__ function defined in AttentionControl Class 
            # forward() and between_steps() are executed in __call__
            if controller.flag:
                attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        #print(net_.__class__.__name__)
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = unet.named_children()
    # sub_nets: <module name, params> of unet model
    # filter of sub_nets, substitute the default forward with defined ca_forward
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    # count the total cross attn module layers: 32 in this repo 
    # half self attn, half text-conditional
    # down: 12 layers, 8 heads each
    # up  : 18 layers, 8 heads each
    # mid : 2  layers, 8 heads each
    controller.num_att_layers = cross_att_count
    #exit()
    