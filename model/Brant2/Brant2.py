import torch
from torch import nn, optim
from argparse import Namespace
from einops import rearrange

from data_process.data_info import data_info_dict
from model.Brant2.model.encoder import Brant2Encoder, Encoder
from model.Brant2.model.utils import Embedding
from model.pre_cnn import ConvNet


class Brant2_Trainer:
    def __init__(self, args: Namespace):
        return

    @staticmethod
    def set_config(args: Namespace):
        args.load_pretrained = True

        args.seq_len = 16
        args.d_model = args.final_dim = 2560

        args.time_n_layers = 8
        args.channel_n_layers = 2
        args.n_heads = 8

        args.norm_eps = 1e-7
        args.ff_hidden = args.d_model * 3
        args.drop_prob = 0.1
        args.learnable_mask = False
        args.mask_ratio = 0.4

        return args

    @staticmethod
    def clsf_loss_func(args):
        ce_weight = [1 for _ in range(args.n_class)]
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
                {'params': list(model.encoder.parameters()), 'lr': args.model_lr},
                {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
        ],
            betas=(0.9, 0.95), eps=1e-5,
        )

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 20], gamma=0.1)







class Brant2(nn.Module):
    def __init__(self, args: Namespace,):
        super(Brant2, self).__init__()

        self.encoder = self.load_pretrained_weights(args=args,
                                                    state_dict_path='/data/yzz/Brant-2/model_ckpt/2560-8-2/brant2_20.pt',
                                                    )

        self.emb = Embedding(args, do_mask=False)
        self.time_encoder = Encoder(args, args.time_n_layers)
        self.channel_encoder = Encoder(args, args.channel_n_layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, psd):
        emb, z = self.encoder(x, psd)
        return emb, z

    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, psd, y = data_packet
        bsz, ch_num, seq_len, patch_len = x.shape

        # emb = model(x, power)               # (bsz, ch_num, seq_len, 1024)
        #
        # emb = torch.mean(emb, dim=2)    # (bsz, ch_num, 1024)

        emb_wei = nn.Parameter(torch.tensor([0.5, 0.5], device=args.gpu_id), requires_grad=True)
        softmax = nn.Softmax(dim=-1)

        emb, z = model(x, psd.float())
        normalized_wei = softmax(emb_wei)
        weighted_emb = emb[:, :, 0] * normalized_wei[0] + emb[:, :, 1] * normalized_wei[1]
        emb = weighted_emb

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
    def load_pretrained_weights(args, state_dict_path,):
        brant2 = Brant2Encoder(args, do_mask=False).to(args.gpu_id)
        if args.load_pretrained:
            brant2_state_dict = torch.load(state_dict_path, map_location=f'cuda:{args.gpu_id}')
            brant2.load_state_dict(brant2_state_dict)

        return brant2

