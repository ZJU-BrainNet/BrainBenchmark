import torch
from torch import nn as nn
from argparse import Namespace


# from model.OneFitsAll.models.masked_tf_model import MaskedTFModel
# from model.OneFitsAll.pre_cnn import ConvNet
from data_process.data_info import data_info_dict
from model.GPT4TS.models.gpt4ts import gpt4ts
from model.pre_cnn import ConvNet


class GPT4TS_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.final_dim =  768
        return args

    @staticmethod
    def clsf_loss_func(args):
        ce_weight = [0.1, 1]
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.cnn.parameters()), 'lr': args.clsf_lr},  # a large lr
            {'params': filter(lambda p: p.requires_grad, model.model.parameters()), 'lr': args.model_lr},
            {'params': filter(lambda p: p.requires_grad, model.cnn1.parameters()), 'lr': args.clsf_lr},  # a large lr
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        ],
            betas=(0.9, 0.95), eps=1e-5,
        )







class GPT4TS(nn.Module):

    def __init__(self, args: Namespace, ):
        super(GPT4TS, self).__init__()
        self.ch_num = 1 # args.channel_num
        self.d_model = 768

        self.cnn = ConvNet(num_inputs=1, num_channels=[self.d_model])
        self.model = gpt4ts(args)
        self.cnn1 = ConvNet(num_inputs=self.ch_num, num_channels=[self.d_model])

    def forward(self, x):
        batch_size, ch_num, seq_len, patch_len = x.shape

        x = x.reshape(batch_size*ch_num*seq_len, 1, patch_len)

        # 将 x_enc 转换为与权重相同的类型
        # x_enc = x_enc.to(self.device, dtype=torch.float)
        emb = self.cnn(x)
        emb = torch.mean(emb, dim=-1)
        emb = emb.reshape(batch_size, ch_num, seq_len, -1)

        # emb = self.norm(emb)    # [batch_size*ch_num, seq_len, d_model]
        #
        # emb += self.positional_encoding
        #
        # outputs = self.gpt2(inputs_embeds=emb).last_hidden_state            # [batch_size*ch_num, seq_len, d_model]
        # outputs = self.act(outputs).reshape(batch_size*ch_num, -1)  # [batch_size*ch_num, seq_len*d_model]
        # outputs = self.ln_proj(outputs)                                     # [batch_size*ch_num, seq_len*d_model]
        #
        # outputs = outputs.reshape(batch_size, ch_num, -1)

        # outputs = self.cnn1(outputs)  # [batch_size, d_model, ???]
        # outputs = torch.mean(outputs, dim=-1) # [batch_size, d_model]


        # outputs = self.out_layer(outputs)  # output.shape: [batch_size, num_class]

        outputs = self.model(emb)       # (bsz*ch_num, seq_len*768)

        # outputs = outputs.reshape(batch_size, ch_num, -1)
        # outputs = self.cnn1(outputs)  # [batch_size, d_model, ???]
        # outputs = torch.mean(outputs, dim=-1) # [batch_size, d_model]

        outputs = outputs.reshape(batch_size, ch_num, seq_len, -1)
        outputs = torch.mean(outputs, dim=2) # [batch_size, ch_num, d_model]


        return outputs


    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, seq_len, patch_len = x.shape

        emb = model(x)

        if data_info_dict[args.dataset]['label_level'] == 'channel_level':
            # y: (bsz, ch_num)
            y = y.reshape(-1)
            emb = emb.reshape(bsz * ch_num, 1, -1)  # (bsz*ch_num, fake_ch_num=1, emb_dim) to clsf
        else:
            # y: (bsz, )
            emb = emb.reshape(bsz, ch_num, -1)

        logit = clsf(emb)  # use the unified clsf

        if args.run_mode != 'test':
            loss = loss_func(logit, y)
            return loss, logit, y
        else:
            return logit, y


