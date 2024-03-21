# from models.embed import DataEmbedding, DataEmbedding_wo_time
from argparse import Namespace

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import GPT2Config, GPT2Model
from einops import rearrange
from model.GPT4TS.models.embed import DataEmbedding

from data_process.data_info import data_info_dict
from model.pre_cnn import ConvNet


class gpt4ts(nn.Module):

    def __init__(self, args: Namespace, ):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.device = args.gpu_id
        self.seq_len = args.seq_len
        self.max_len = args.seq_len
        self.patch_size = args.patch_len
        self.stride = 1
        self.gpt_layers = 6
        # self.feat_dim = data.feature_df.shape[1]
        self.feat_dim = 1
        self.num_class = args.n_class
        self.d_model = 768

        # # self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        # self.patch_num = self.seq_len
        #
        # self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        # self.patch_num += 1
        # self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, 768)
        # # self.enc_embedding = DataEmbedding(self.patch_size, 768)    # c_in是里面卷积的输入通道数

        local_model_path = "/home/ggf/code/Brant2-down/onefitsall/gpt2"

        config = GPT2Config.from_pretrained(local_model_path)
        self.gpt2 = GPT2Model.from_pretrained(local_model_path, config=config)
        # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)

        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # device = torch.device('cuda:{}'.format(args.gpu_id))
        # self.gpt2.to(device=device)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        # self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)

        self.ln_proj = nn.LayerNorm(self.d_model * self.seq_len)
        # self.out_layer = nn.Linear(self.d_model * self.patch_num, self.num_classes)
         # init
        # batch_size,ch_num,seq_len,patch_len = data.shape
        # self.cnn = ConvNet(num_inputs=1, num_channels=[self.d_model]).to(device)
        self.norm = nn.LayerNorm(self.d_model)
        self.out_layer = nn.Linear(self.d_model, self.num_class)
        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, self.d_model), requires_grad=True)

    def forward(self, x):
        batch_size, ch_num, seq_len, patch_len = x.shape

        x = x.reshape(batch_size*ch_num, seq_len, -1)

        # input_x = rearrange(x, 'b l m -> b m l')
        # input_x = self.padding_patch_layer(input_x)
        # # input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # input_x = input_x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        # input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        #
        # outputs = self.enc_embedding(input_x, None)
        #
        # outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state
        #
        # outputs = self.act(outputs).reshape(batch_size*ch_num, -1)
        # outputs = self.ln_proj(outputs)


        emb = self.norm(x)    # [batch_size*ch_num, seq_len, d_model]

        emb += self.positional_encoding

        outputs = self.gpt2(inputs_embeds=emb).last_hidden_state            # [batch_size*ch_num, seq_len, d_model]
        outputs = self.act(outputs).reshape(batch_size*ch_num, -1)  # [batch_size*ch_num, seq_len*d_model]
        outputs = self.ln_proj(outputs)                                     # [batch_size*ch_num, seq_len*d_model]
        return outputs
