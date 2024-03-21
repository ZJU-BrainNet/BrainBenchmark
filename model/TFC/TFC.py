from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from argparse import Namespace

from data_process.data_info import data_info_dict
from model.TFC.loss import NTXentLoss_poly



class TFC_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.Context_Cont_temperature = 0.2
        args.Context_Cont_use_cosine_similarity = True
        args.augmentation_max_seg = 12
        args.augmentation_jitter_ratio = 2
        args.augmentation_jitter_scale_ratio = 1.5
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
            betas=(0.9, 0.95), eps=1e-5,
        )


class TFC(nn.Module):
    def __init__(self, args: Namespace):
        super(TFC, self).__init__()
        TSlength_aligned = args.patch_len
        seq_len = args.seq_len
        final_dim = args.final_dim

        encoder_layers_t = TransformerEncoderLayer(TSlength_aligned, dim_feedforward=2 * TSlength_aligned, nhead=2, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(TSlength_aligned * seq_len, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, final_dim // 2)
        )

        encoder_layers_f = TransformerEncoderLayer(TSlength_aligned, dim_feedforward=2 * TSlength_aligned, nhead=2, )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(TSlength_aligned * seq_len, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(),
            nn.Linear(final_dim, final_dim // 2)
        )

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        # x: (bsz, ch_num, seq_len, patch_len)
        # TFC model requires input as (bsz, seq_len, patch_len)
        device = next(model.parameters()).device

        if args.run_mode == "unsupervised":
            x, aug1_x, f, aug1_f = data_packet

            bsz, ch_num, seq_len, patch_len = x.shape
            x = x.reshape(bsz * ch_num, seq_len, patch_len).float()
            aug1_x = aug1_x.reshape(bsz * ch_num, seq_len, patch_len).float()
            f = f.reshape(bsz * ch_num, seq_len, patch_len).float()
            aug1_f = aug1_f.reshape(bsz * ch_num, seq_len, patch_len).float()

            """Produce embeddings"""
            h_t, z_t, h_f, z_f = model(x, f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1_x, aug1_f)

            """Compute Pre-train loss"""
            """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
            nt_xent_criterion = NTXentLoss_poly(device, bsz * ch_num, args.Context_Cont_temperature,
                                                args.Context_Cont_use_cosine_similarity)  # device, 128, 0.2, True

            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            l_TF = nt_xent_criterion(z_t, z_f)  # this is the initial version of TF loss

            l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug,
                                                                                                                z_f_aug)
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

            lam = 0.2
            loss = lam * (loss_t + loss_f) + l_TF

            return loss

        elif args.run_mode == "finetune" or args.run_mode == "test":
            x, y, aug1_x, f, aug1_f = data_packet

            bsz, ch_num, seq_len, patch_len = x.shape
            x = x.reshape(bsz * ch_num, seq_len, patch_len).float()
            aug1_x = aug1_x.reshape(bsz * ch_num, seq_len, patch_len).float()
            f = f.reshape(bsz * ch_num, seq_len, patch_len).float()
            aug1_f = aug1_f.reshape(bsz * ch_num, seq_len, patch_len).float()

            """Produce embeddings"""
            h_t, z_t, h_f, z_f = model(x, f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1_x, aug1_f)

            nt_xent_criterion = NTXentLoss_poly(device, bsz * ch_num, args.Context_Cont_temperature,
                                                args.Context_Cont_use_cosine_similarity)
            # print("ht_shape:",h_t.shape,h_t_aug.shape)
            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            l_TF = nt_xent_criterion(z_t, z_f)

            l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), \
                            nt_xent_criterion(z_t_aug, z_f_aug)
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)  #

            fea_concat = torch.cat((z_t, z_f), dim=1)  # (bsz*ch_num, emb_dim)

            if data_info_dict[args.dataset]['label_level'] == 'channel_level':
                # y: (bsz, ch_num)
                y = y.reshape(-1)
                fea_concat = fea_concat.reshape(bsz * ch_num, 1, -1)  # (bsz*ch_num, fake_ch_num=1, emb_dim) to clsf
            else:
                # y: (bsz, )
                fea_concat = fea_concat.reshape(bsz, ch_num, -1)
            logit = clsf(fea_concat)  # use the unified clsf

            if args.run_mode != 'test':
                loss_p = loss_func(logit, y)
                lam = 0.1
                loss = loss_p + l_TF + lam * (loss_t + loss_f)
                return loss, logit, y
            else:
                return logit, y

        else: raise NotImplementedError(f'Undefined running mode {args.run_mode}')



