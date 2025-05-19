import torch
import torch.nn as nn

from utils.attention import AttentionLayer, FullAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers import GPT2Tokenizer


class IOHFuseLM(nn.Module):
    
    def __init__(self, configs, device):
        super(IOHFuseLM, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = self.patch_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_num = ((configs.seq_len - self.patch_size) // self.stride) + 1
        
        self.cnt = 0
        self.max_txt_len = configs.max_txt_len

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 

        
        self.mask_rate = 0
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(configs.tokenizer_path)
        
        if configs.is_gpt:
            self.gpt2 = GPT2Model.from_pretrained(configs.tokenizer_path)
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.out_layer = nn.Linear(configs.d_model * (self.max_txt_len + self.patch_num), configs.pred_len) #* self.max_token_num
        
        self.att = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads)
        
        self.pad_token = nn.Parameter(torch.randn(1, 1, configs.d_model), requires_grad=True)
        
        if configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

    def generate_ts_token(self, x_inp, stride):
        
        x_inp = x_inp.unfold(dimension=-1, size=self.patch_size, step=stride)

        b, f, p, h = x_inp.shape
        x_inp = x_inp.reshape(b * f, p, h)
        x_embed = self.in_layer(x_inp)

        return self.dropout(x_embed)

    def forward(self, x, instruct):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')
        x_token = self.generate_ts_token(x, self.stride)
        
        if len(instruct) > 0:
            instruct_encoding = self.tokenizer(
                instruct,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
                return_attention_mask=True
            )
            instruct_contents = instruct_encoding.input_ids.to(x.device)
            instruct_mask = instruct_encoding.attention_mask.to(x.device)
            instruct_embed = self.gpt2.wte(instruct_contents)
            self.att = self.att.to(x.device)
            ts_mask = torch.ones_like(x_token)
            instruct_att, _ = self.att(instruct_embed, x_token, x_token, instruct_mask, ts_mask)
            inputs_embeds = torch.cat((instruct_att, x_token), dim=1)
        else:
            inputs_embeds = x_token
        
        if self.is_gpt:
            x_enc = self.gpt2(inputs_embeds=inputs_embeds).last_hidden_state
        
        x_dec = torch.reshape(
            x_enc, (-1, M, x_enc.shape[-2], x_enc.shape[-1]))
        x_dec = x_dec.permute(0, 1, 3, 2)
            
        outputs = self.out_layer(x_dec.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        return outputs
