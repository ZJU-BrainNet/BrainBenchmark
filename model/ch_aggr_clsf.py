import torch
from torch import nn
from argparse import Namespace


class ChannelCNN(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(ChannelCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=args.cnn_in_channels, out_channels=args.final_dim // 2, kernel_size=args.cnn_kernel_size),
            nn.BatchNorm1d(args.final_dim // 2),
            nn.Conv1d(in_channels=args.final_dim // 2, out_channels=args.final_dim, kernel_size=args.cnn_kernel_size),
            nn.BatchNorm1d(args.final_dim),
            nn.Conv1d(in_channels=args.final_dim, out_channels=args.final_dim, kernel_size=args.cnn_kernel_size),
        )

    def forward(self, x):
        bsz, ch_num, emb_dim = x.shape
        if ch_num != 1:
            x = self.cnn(x)                     # (bsz, final_dim, --)
            emb = torch.mean(x, dim=-1)   # (bsz, final_dim)
        else:
            emb = torch.squeeze(x, 1) # (bsz, emb_dim)
        return emb


class ChannelAggrClsf(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(ChannelAggrClsf, self).__init__()

        # channel aggregation
        self.cnn = ChannelCNN(args)

        # classifier
        self.d_model = args.final_dim
        self.out_dim = args.n_class

        self.hidden = 2 * (self.d_model + self.out_dim) // 3
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (bsz, ch_num, seq_len, patch_len)
        x = self.cnn(x)     # x: (bsz, final_dim)

        logit = self.softmax(self.head(x))

        return logit  # (bsz, num_class)
