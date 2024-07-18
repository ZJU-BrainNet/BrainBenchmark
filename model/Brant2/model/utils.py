from argparse import Namespace

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class ConvBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dropout):
        super(ConvBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.maxpool1 = nn.MaxPool1d(4, 2)

        self.net = nn.Sequential(self.conv1, self.relu1, self.maxpool1, self.dropout1)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, dropout=0.2):
        super(ConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            kernel_size = 2 ** (i + 2)
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ConvBlock(in_channels, out_channels, kernel_size, stride=1, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # multiply with Value
        v = score @ v

        return v, score


class Embedding(nn.Module):
    def __init__(self, args: Namespace, do_mask):
        super(Embedding, self).__init__()
        self.mask_ratio = args.mask_ratio
        self.do_mask = do_mask

        out_dim = args.d_model // 2
        # self.tcn = TemporalConvNet(num_inputs=1, num_channels=[out_dim // 2, out_dim])
        self.tcn = ConvNet(num_inputs=1, num_channels=[out_dim // 2, out_dim])
        self.proj1 = nn.Linear(out_dim, out_dim)
        self.norm1 = LayerNorm(d_model=out_dim)

        self.cnn = ConvNet(num_inputs=1, num_channels=[out_dim // 2, out_dim])
        self.proj2 = nn.Linear(out_dim, out_dim)
        self.norm2 = LayerNorm(d_model=out_dim)
        # learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(args.seq_len, args.d_model), requires_grad=True)
        self.mask_encoding = nn.Parameter(torch.randn(args.d_model), requires_grad=True) if args.learnable_mask else \
                             nn.Parameter(torch.zeros(args.d_model), requires_grad=False)

    def forward(self, x, psd):
        # x: [batch_size, ch_num, seq_len, patch_len]
        batch_size, ch_num, seq_len, patch_len = x.shape

        x = x.reshape(batch_size*ch_num*seq_len, 1, patch_len)
        t_emb = self.tcn(x)
        t_emb = torch.mean(t_emb, dim=-1).reshape(batch_size, ch_num*seq_len, -1)
        t_emb = self.norm1(self.proj1(t_emb))

        psd = psd.reshape(batch_size*ch_num*seq_len, 1, -1)
        psd_emb = self.cnn(psd)
        psd_emb = torch.mean(psd_emb, dim=-1).reshape(batch_size, ch_num*seq_len, -1)
        psd_emb = self.norm2(self.proj2(psd_emb))
        emb = torch.concat([t_emb, psd_emb], dim=-1)

        # mask
        if self.do_mask:
            mask_num = int(ch_num*seq_len*self.mask_ratio)
            mask_pos = np.random.permutation(ch_num*seq_len)[:mask_num]
            emb[:, mask_pos, :] = self.mask_encoding

        emb = emb.reshape(batch_size*ch_num, seq_len, -1)
        emb += self.positional_encoding
        emb = torch.unsqueeze(emb, dim=1)
        emb = torch.tile(emb, (1, 2, 1, 1))  # [bsz, 2, seq_len, d_model]
        return emb


class PredictHead(nn.Module):
    def __init__(self, args, patch_len, forecast_len):
        super(PredictHead, self).__init__()
        self.d_model = args.d_model
        self.mask_head = nn.Linear(self.d_model, patch_len)
        self.forecast_head = nn.Linear(self.d_model, forecast_len)

    def forward(self, x):
        # x: [bsz, ch_num, 2, seq_len, d_model]
        mask_pred = self.mask_head(x[:, :, 0])
        forecast_pred = self.forecast_head(torch.mean(x[:, :, 1], dim=-2))

        return mask_pred, forecast_pred


class ClassificationHead(nn.Module):
    def __init__(self, args, num_class, num_channel):
        super(ClassificationHead, self).__init__()
        self.d_model = args.d_model // 4 if num_channel > 1 else args.d_model
        self.cnn = ConvNet(num_inputs=num_channel, num_channels=[self.d_model])
        self.head = nn.Linear(self.d_model, num_class)

    def forward(self, x):
        bsz, ch_num, d_model = x.shape
        if ch_num > 1:
            x = self.cnn(x)
            x = torch.mean(x, dim=-1)
        else:
            x = torch.squeeze(x, dim=1)
        logit = self.head(x)

        return logit  # [bsz, num_class]


class LinearHead(nn.Module):
    def __init__(self, args, out_dim, hidden):
        super(LinearHead, self).__init__()
        if hidden:
            hidden_dim = 2*(args.d_model+out_dim) // 3
            self.head = nn.Sequential(
                nn.Linear(args.d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.drop_prob),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(args.drop_prob),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            self.head = nn.Linear(args.d_model, out_dim)

    def forward(self, x):
        return self.head(x)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.head_dim = args.d_model // args.n_heads
        self.n_heads = args.n_heads

        self.wq = nn.Linear(
            args.d_model,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.d_model,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.d_model,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.d_model,
            args.n_heads * self.head_dim,
            bias=False,
        )

    def forward(self, x):
        bsz_m_ff, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz_m_ff, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz_m_ff, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz_m_ff, seqlen, self.n_heads, self.head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz_m_ff, seqlen, -1)

        return self.wo(output)






