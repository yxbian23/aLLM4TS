import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import AutoModel, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
from layers.RevIN import RevIN

class Model(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        self.is_llm = configs.is_llm
        self.pretrain = configs.pretrain

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_len) // self.stride + 1

        self.revin = configs.revin
        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=configs.subtract_last)
        

        if configs.is_llm:
            if configs.pretrain:
                if "gpt2" in configs.llm:
                    self.gpt = GPT2Model.from_pretrained(configs.llm, output_attentions=True, output_hidden_states=True)  
                elif "llama" in configs.llm:
                    self.gpt = LlamaModel.from_pretrained(configs.llm, output_attentions=True, output_hidden_states=True)  
                else:
                    raise NotImplementedError
            else:
                print("------------------no pretrain------------------")
                if "gpt2" in configs.llm:
                    self.gpt = GPT2Model(GPT2Config())
                elif "llama" in configs.llm:
                    self.gpt = LlamaModel(LlamaConfig())
                else:
                    raise NotImplementedError
            if "gpt2" in configs.llm:
                self.gpt.h = self.gpt.h[:configs.llm_layers]
                print("gpt2 = {}".format(self.gpt))
            elif "llama" in configs.llm:
                self.gpt.layers = self.gpt.layers[:configs.llm_layers]
                print("llama2 = {}".format(self.gpt))
            else:
                raise NotImplementedError
        
        self.in_layer = nn.Linear(self.patch_len, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model, self.patch_len)


        if configs.freeze:
            if configs.c_pt:
                layers_train = configs.pt_layers.split('_')
            elif configs.sft:
                layers_train = configs.sft_layers.split('_')
            else:
                raise NotImplementedError

            for i, (name, param) in enumerate(self.gpt.named_parameters()):
                tag = 0
                for layer_train in layers_train:
                    if layer_train in name:
                        tag = 1
                if tag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
        for layer in (self.gpt, self.in_layer, self.out_layer, self.revin_layer):
            layer.train()
        

    def forward(self, x):
        # x: ts origin data, (batch, length, channel)
        B, L, M = x.shape
        if self.revin: 
            x = self.revin_layer(x, 'norm')

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        # llm encode Temporal Data
        ts_inputs = self.in_layer(x) # (b m) n p -> (b m) n d_model

        # llm embedding learning
        outputs = self.gpt(inputs_embeds=ts_inputs).last_hidden_state

        # Project Back to the Patch from the 'token' 
        outputs = self.out_layer(outputs)
        outputs = rearrange(outputs, '(b m) n p -> b (n p) m', b=B, m=M)
        if self.revin: 
            outputs = self.revin_layer(outputs, 'denorm')
        outputs = rearrange(outputs, 'b (n p) m -> b m n p', n=self.patch_num ,p=self.patch_len)

        return outputs
