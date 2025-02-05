from torch import nn
import torch
from argparse import Namespace

from data_process.data_info import data_info_dict
from model.SimMTM.loss import ContrastiveWeight, AggregationRebuild, AutomaticWeightedLoss
from model.SimMTM.augmentations import data_transform_masked4cl



class SimMTM_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.kernel_size = 8
        args.stride = 1
        args.dropout = 0.35
        args.temperature = 0.2
        args.positive_nums = 3
        args.masking_ratio = 0.5
        args.lm = 3

        return args

    @staticmethod
    def clsf_loss_func(args):
        return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': list(model.parameters()), 'lr': args.lr},
            {'params': list(clsf.parameters()), 'lr': args.lr},
        ],
            betas=(0.9, 0.999), eps=1e-8,
        )


class SimMTM(nn.Module):
    def __init__(self, args: Namespace):
        super(SimMTM, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=args.kernel_size,
                      stride=args.stride, bias=False, padding=(args.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(args.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, args.final_dim, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(args.final_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # self.dense = nn.Sequential(
        #     nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128)
        # )

        self.dense = nn.Sequential(
            nn.Linear(args.final_dim, args.final_dim),
            nn.BatchNorm1d(args.final_dim),
            nn.ReLU(),
            nn.Linear(args.final_dim, args.final_dim)
        )

        if args.run_mode == 'unsupervised':
            self.awl = AutomaticWeightedLoss(2)
            self.contrastive = ContrastiveWeight(args)
            self.aggregation = AggregationRebuild(args)
            self.head = nn.Linear(args.final_dim, args.seq_len*args.patch_len)
            self.mse = torch.nn.MSELoss()

    def forward(self, x_in_t, pretrain=False):

        # SimTMT model can only process single-channel data, and channel merging is required
        bsz, ch_num = x_in_t.shape[0], x_in_t.shape[1]
        x_in_t = x_in_t.reshape(bsz * ch_num, -1)
        x_in_t = torch.unsqueeze(x_in_t, dim=1) 
        
        if pretrain:
            x = self.conv_block1(x_in_t)
            x = self.conv_block2(x)
            x = self.conv_block3(x)

            # h = x.reshape(x.shape[0], -1)
            h = torch.mean(x, dim=-1)
            z = self.dense(h)

            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(z)
            rebuild_weight_matrix, agg_x = self.aggregation(similarity_matrix, x)
            # pred_x = self.head(agg_x.reshape(agg_x.size(0), -1))
            pred_x = self.head(torch.mean(agg_x, dim=-1))

            loss_rb = self.mse(pred_x, x_in_t.reshape(x_in_t.size(0), -1).detach())
            loss = self.awl(loss_cl, loss_rb)

            return loss
        else:
            # x_in_t = self.cnn(x_in_t) # x_in_t: (batch size*ch_num, out_dim, seg_len')
            # x_in_t = torch.mean(x_in_t, -1) # x_in_t: (batch size*ch_num, out_dim)
            # x_in_t = x_in_t.reshape(bsz,ch_num,-1)
            # x_in_t: (batch size,ch_num, out_dim)

            x = self.conv_block1(x_in_t)
            # x: (bsz*ch_num,32,???)
            x = self.conv_block2(x)
            # x: (bsz*ch_num,64,???)
            x = self.conv_block3(x)
            # x: (bsz*ch_num,final_out_channels,???)

            h = torch.mean(x, dim=-1)  # (bsz*ch_num, final_out_channels)
            h = h.reshape(bsz, ch_num, -1)

            return h


    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        bsz, ch_num, seq_len, patch_len = x.shape

        if args.run_mode == "unsupervised":
            x = torch.reshape(x, (bsz, ch_num, -1))
            x_masked_m, mask = data_transform_masked4cl(x, args.masking_ratio, args.lm, args.positive_nums)
            x_masked_om = torch.cat([x, x_masked_m], 0).to(args.gpu_id)

            loss = model(x_masked_om, pretrain=True)
            return loss

        elif args.run_mode == "finetune" or args.run_mode == "test":
            emb = model(x)                     # (bsz, ch_num, 256)

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

        else: raise NotImplementedError(f'Undefined running mode {args.run_mode}')


