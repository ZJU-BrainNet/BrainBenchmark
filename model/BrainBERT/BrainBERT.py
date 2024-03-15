import torch
from omegaconf import OmegaConf
from torch import nn
from argparse import Namespace

from data_process.data_info import data_info_dict
from model.BrainBERT.models.masked_tf_model import MaskedTFModel
from model.BrainBERT.pre_cnn import ConvNet


class BrainBERT_Trainer:
    def __init__(self, args: Namespace):
        super(BrainBERT_Trainer, self).__init__()

    @staticmethod
    def set_config(parser):
        group_model = parser.add_argument_group('Model')
        group_model.add_argument('--final_dim', type=int, default=768,
                                 help="The dim of final representations.")
        args = parser.parse_args()
        return args

    @staticmethod
    def clsf_loss_func(args):
        return nn.CrossEntropyLoss(torch.tensor([0.4, 1], dtype=torch.float32, device=torch.device(args.gpu_id)))

    @staticmethod
    def optimizer(args, model, clsf):
        return torch.optim.AdamW([
            {'params': filter(lambda p: p.requires_grad, model.cnn.parameters()), 'lr': args.clsf_lr},  # a large lr
            {'params': filter(lambda p: p.requires_grad, model.enc.parameters()), 'lr': args.model_lr},
            {'params': list(clsf.parameters()), 'lr': args.clsf_lr}
        ],
            betas=(0.9, 0.95), eps=1e-5,
        )


# def set_BrainBERT_config(parser):
#     group_model = parser.add_argument_group('Model')
#     group_model.add_argument('--final_dim', type=int, default=768,
#                              help="The dim of final representations.")
#     args = parser.parse_args()
#     return args
#
#
# # @staticmethod
# def load_pretrained_weights_BrainBERT(args, ):
#     def _build_model(cfg, gpu_id):
#         ckpt_path = cfg.upstream_ckpt
#         init_state = torch.load(ckpt_path, map_location=f'cuda:{gpu_id}')
#         upstream_cfg = init_state["model_cfg"]
#
#         # model = models.build_model(upstream_cfg)
#         model = MaskedTFModel()
#         model.build_model(upstream_cfg, )
#
#         model.load_state_dict(init_state['model'])
#         return model
#
#     cfg = OmegaConf.create({"upstream_ckpt": '/data/yzz/Brant-2/baseline_ckpt/BrainBERT/stft_large_pretrained.pth'})
#     model = _build_model(cfg, args.gpu_id).to(args.gpu_id)
#     return model
#
# # @staticmethod
# def clsf_loss_func(args):
#     return nn.CrossEntropyLoss(torch.tensor([0.4, 1], dtype=torch.float32, device=torch.device(args.gpu_id)))



# def adapt_to_input_size_BrainBERT(args, x_list):
#     new_x_list = []
#     for x in x_list:
#         seq_num, ch_num, seq_len, patch_len = x.shape
#         x = x.reshape(seq_num, ch_num, -1)
#
#         tgt_patch_len = 40
#         new_len = (seq_len * patch_len) // tgt_patch_len * tgt_patch_len
#         x = x[:, :, :new_len].reshape(seq_num, ch_num, new_len//tgt_patch_len, tgt_patch_len)
#         new_x_list.append(x)
#
#     return new_x_list







class BrainBERT(nn.Module):
    def __init__(self, args: Namespace,):
        super(BrainBERT, self).__init__()
        in_patch_len = 40
        self.cnn = ConvNet(num_inputs=1, num_channels=[in_patch_len])

        self.enc = self.load_pretrained_weights(args)

    def forward(self, x):
        bsz, ch_num, seq_len, patch_len = x.shape
        x = x.reshape(bsz*ch_num*seq_len, 1, patch_len)
        emb = self.cnn(x)
        emb = torch.mean(emb, dim=-1).reshape(bsz*ch_num, seq_len, -1)
        emb = self.enc.forward(emb, intermediate_rep=True)
        emb = emb.reshape(bsz, ch_num, seq_len, -1)
        emb = torch.mean(emb, dim=2)
        return emb

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

    @staticmethod
    def load_pretrained_weights(args, ):
        def _build_model(cfg, gpu_id):
            ckpt_path = cfg.upstream_ckpt
            init_state = torch.load(ckpt_path, map_location=f'cuda:{gpu_id}')
            upstream_cfg = init_state["model_cfg"]

            # model = models.build_model(upstream_cfg)
            model = MaskedTFModel()
            model.build_model(upstream_cfg, )

            model.load_state_dict(init_state['model'])
            return model

        cfg = OmegaConf.create({"upstream_ckpt": '/data/yzz/Brant-2/baseline_ckpt/BrainBERT/stft_large_pretrained.pth'})
        model = _build_model(cfg, args.gpu_id).to(args.gpu_id)
        return model

