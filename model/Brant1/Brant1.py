import torch
from torch import nn, optim
from argparse import Namespace
from einops import rearrange

from data_process.data_info import data_info_dict
from model.Brant1.models import TimeEncoder, ChannelEncoder
from model.pre_cnn import ConvNet
from utils.config import config

class Brant1_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.load_pretrained = True
        args.unwrap_ddp = True
        args.freeze_encoder = False
        args.final_dim = 1024
        args.start_epo_idx = 21
        # args.power_save_path = f'{config["gene_data_path"]}{args.data_id}/power.npy'
        return args

    @staticmethod
    def clsf_loss_func(args):
        if args.n_class != 2:
            import numpy as np
            ce_weight = [0.8 for _ in range(args.n_class - 1)]
            ce_weight.append(1.0)
        else:
            ce_weight = [0.8, 1]
        print(f'CrossEntropy loss weight = {ce_weight} = {ce_weight[1]/ce_weight[0]:.2f}')
        return nn.CrossEntropyLoss(torch.tensor(ce_weight, dtype=torch.float32, device=torch.device(args.gpu_id)))
        # return nn.CrossEntropyLoss()

    @staticmethod
    def optimizer(args, model, clsf):
        # return torch.optim.Adam([
        #     {'params': list(model.cnn.parameters()), 'lr': args.clsf_lr},
        #     {'params': list(model.encoder_t.parameters()), 'lr': args.model_lr},
        #     {'params': list(model.encoder_ch.parameters()), 'lr': args.model_lr},
        #     {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        # ],
        #     betas=(0.9, 0.99), eps=1e-8,
        # )
        return torch.optim.AdamW([
                {'params': list(model.cnn.parameters()), 'lr': args.clsf_lr},
                {'params': list(model.encoder_t.parameters()), 'lr': args.model_lr},
                {'params': list(model.encoder_ch.parameters()), 'lr': args.model_lr},
                {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.99), eps=1e-6,
        )

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)







class Brant1(nn.Module):
    def __init__(self, args: Namespace,):
        super(Brant1, self).__init__()

        self.cnn = ConvNet(num_inputs=1, num_channels=[512, 22500])
        self.encoder_t, self.encoder_ch = self.load_pretrained_weights(args, scale='100M')  # '100M' '500M'

    def forward(self, x, power):
        bat_size, ch_num, seq_len, patch_len = x.shape

        if seq_len!=15 or patch_len!=1500:
            x = torch.unsqueeze(x, dim=2)  # (bsz, ch_num, 1, seq_len, patch_len)
            x = rearrange(x, 'b c f s p -> (b c) f (s p)')
            x = self.cnn(x)                       # (bsz*ch_num, 22500, xx)
            x = torch.mean(x, dim=-1)     # (bsz*ch_num, 22500,)
            x = x.reshape(bat_size, ch_num, 15, 1500)

            bat_size, ch_num, seq_len, patch_len = x.shape

        # time_z = self.encoder_t(mask=None, data=x, power=power, need_mask=False)  # time_z.shape: new_bat * ch_num, seq_len, d_model
        time_z = self.encoder_t(mask=None, data=x, power=None, need_mask=False, mask_by_ch=False, rand_mask=True, mask_len=None, use_power=False)

        if ch_num==1:
            emb = time_z
            emb = emb.reshape(bat_size, ch_num, seq_len, -1)

        else:
            _, _, d_model = time_z.shape
            time_z = time_z.reshape(bat_size, ch_num, seq_len, d_model)  # time_z.shape: new_bat, ch_num, seq_len, d_model
            time_z = torch.transpose(time_z, 1, 2)          # time_z.shape: new_bat, seq_len, ch_num, d_model
            time_z = time_z.reshape(bat_size * seq_len, ch_num, d_model)  # time_z.shape: new_bat*seq_len, ch_num, d_model

            emb, _ = self.encoder_ch(time_z)  # emb.shape: new_bat*seq_len, ch_num, d_model

            emb = emb.reshape(bat_size, seq_len, ch_num, d_model)
            emb = torch.transpose(emb, 1, 2)    #  bat_size, ch_num, seq_len, d_model

        return emb

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, power, y = data_packet
        bsz, ch_num, seq_len, patch_len = x.shape

        emb = model(x, power)               # (bsz, ch_num, seq_len, 1024)

        emb = torch.mean(emb, dim=2)    # (bsz, ch_num, 1024)

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

    @staticmethod
    def load_pretrained_weights(args, scale):
        from collections import OrderedDict
        def unwrap_ddp(state_dict: OrderedDict):
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_state_dict[key[7:]] = value
            return new_state_dict

        if scale == '100M':
            final_dim, dim_ffd, t_n_layer, ch_n_layer, dir_name = 1024, 2048, 8, 4, 'set1024_8_4'
        elif scale == '500M':
            final_dim, dim_ffd, t_n_layer, ch_n_layer, dir_name = 2048, 3072, 12, 5, 'set2048_12_5'
        else: raise NotImplementedError()

        encoder_t = TimeEncoder(in_dim=1500,
                                d_model=final_dim,
                                dim_feedforward=dim_ffd,
                                seq_len=15,
                                n_layer=t_n_layer,
                                nhead=16,
                                band_num=8,
                                project_mode='linear',
                                learnable_mask=False).to(args.gpu_id)
        encoder_ch = ChannelEncoder(out_dim=1500,
                                    d_model=final_dim,
                                    dim_feedforward=dim_ffd,
                                    n_layer=ch_n_layer,
                                    nhead=16).to(args.gpu_id)

        # --------- pretrained model loading ---------
        if args.load_pretrained:
            # map_location = {'cuda:%d' % 0 : 'cuda:%d' % args.gpu_id}
            map_location = 'cuda:%d' % args.gpu_id
            t_state_dict  = torch.load(f'{config["Brant1_path"]}/{dir_name}/encoder_ckpt/encoder_t_{args.start_epo_idx}.pt',
                                       map_location=map_location)
            ch_state_dict = torch.load(f'{config["Brant1_path"]}/{dir_name}/encoder_ckpt/encoder_ch_{args.start_epo_idx}.pt',
                                       map_location=map_location)
            # t_state_dict  = torch.load(f'.../Brant1/{dir_name}/encoder_ckpt/encoder_t_{args.start_epo_idx}.pt',
            #                            map_location=map_location)
            # ch_state_dict = torch.load(f'.../Brant1/{dir_name}/encoder_ckpt/encoder_ch_{args.start_epo_idx}.pt',
            #                            map_location=map_location)
            if args.unwrap_ddp:
                t_state_dict = unwrap_ddp(t_state_dict)
                ch_state_dict = unwrap_ddp(ch_state_dict)

            encoder_t.load_state_dict(t_state_dict)
            encoder_ch.load_state_dict(ch_state_dict)
            print('----- Pretrained Models Loaded -----\n')

        if args.freeze_encoder:
            for param in encoder_t.parameters():
                param.requires_grad = False
            for param in encoder_ch.parameters():
                param.requires_grad = False

        return encoder_t, encoder_ch

