from argparse import Namespace

import torch
from torch import nn

from model.Brant2.model.utils import Embedding, MultiHeadAttention, RMSNorm


class Brant2Encoder(nn.Module):
    def __init__(self, args: Namespace, do_mask=False):
        super(Brant2Encoder, self).__init__()

        self.emb = Embedding(args, do_mask=do_mask)
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
        # x: bsz, ch_num, seq_len, patch_len
        bsz, ch_num, seq_len, patch_len = x.shape
        # time encoder
        # x = x.reshape(bsz * ch_num, seq_len, -1)
        # psd = psd.reshape(bsz * ch_num, seq_len, -1)
        input_emb = self.emb(x, psd)
        # t_emb: [bsz*ch_num, 2, seq_len, d_model]
        t_emb = self.time_encoder(input_emb)
        # t_emb: [bsz, ch_num, 2, seq_len, d_model]
        t_emb = t_emb.reshape(bsz, ch_num, 2, seq_len, -1)

        # channel encoder
        # t_emb: [bsz*seq_len, 2, ch_num, d_model]
        t_emb = torch.swapaxes(t_emb, axis0=1, axis1=3).reshape(bsz * seq_len, 2, ch_num, -1)
        # ch_emb: [bsz, seq_len, 2, ch_num, d_model]
        ch_emb = self.channel_encoder(t_emb).reshape(bsz, seq_len, 2, ch_num, -1)
        # z: [bsz, ch_num, 2, seq_len, d_model]
        z = torch.swapaxes(ch_emb, axis0=1, axis1=3)

        # emb: [bsz, ch_num, 2, d_model]
        emb = torch.mean(z, dim=-2)

        return emb, z


class EncoderLayer(nn.Module):
    def __init__(self, args: Namespace):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(args)
        self.norm1 = RMSNorm(dim=args.d_model, eps=args.norm_eps)
        # self.norm1 = LayerNorm(d_model=args.d_model)
        drop_prob = args.drop_prob
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn_mask = nn.Sequential(
            nn.Linear(args.d_model, args.ff_hidden),
            nn.SiLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(args.ff_hidden, args.d_model),
        )
        self.ffn_forecast = nn.Sequential(
            nn.Linear(args.d_model, args.ff_hidden),
            nn.SiLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(args.ff_hidden, args.d_model),
        )
        self.norm2 = RMSNorm(dim=args.d_model, eps=args.norm_eps)

        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        batch_size, ff_head, seq_len, d_model = x.size()

        # compute self attention
        x = x.reshape(batch_size*ff_head, seq_len, d_model)
        _x = x
        x = self.attention(x)

        # add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        #  ffn_x: [bsz, 2, seq_len, d_model]
        x = x.reshape(batch_size, ff_head, seq_len, d_model)
        _x = x
        ffn_mask = self.ffn_mask(x[:, 0]).unsqueeze(dim=1)
        mask_help_forecast = x[:, 0].detach().clone()
        mask_help_forecast.requires_grad = False
        ffn_forecast = self.ffn_forecast(mask_help_forecast + x[:, 1]).unsqueeze(dim=1)
        x = torch.concat([ffn_mask, ffn_forecast], dim=1)

        # add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(self, args: Namespace, n_layers):
        super(Encoder, self).__init__()

        self.enc_layers = nn.ModuleList([EncoderLayer(args) for _ in range(n_layers)])

    def forward(self, emb):

        for layer in self.enc_layers:
            emb = layer(emb)

        return emb
