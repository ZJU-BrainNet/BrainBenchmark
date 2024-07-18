import os.path

import torch
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange


class Brant2_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        self.seq_num, self.ch_num, seq_len, patch_len = x.shape

        self.x = x
        self.y = y

        # reshape to seq_len = 16
        x = rearrange(x, 's c l p -> s c (l p)')
        new_patch_len = seq_len*patch_len // 16
        x = x[:, :, :new_patch_len*16]
        self.x = x.reshape(self.seq_num, self.ch_num, 16, new_patch_len)

        # if os.path.exists(args.power_save_path):
        #     self.power = np.load(args.power_save_path)
        # else:
        #     self.power = self.compute_power(x, fs=256)
        #     np.save(args.power_save_path, self.power)
        _, self.psd = self.periodogram(self.x, fs=256)

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        return self.x  [index, :, :, :], \
               self.psd[index, :, :, :], \
               self.y  [index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)

    @staticmethod
    def periodogram(X: np.ndarray, fs=256, detrend=False, scaling='density'):
        if X.ndim > 2:
            # X = np.squeeze(X)
            ...
        elif X.ndim == 1:
            X = np.expand_dims(X, axis=0)

        if detrend:
            X -= np.mean(X, axis=-1, keepdims=True)

        N = X.shape[-1]
        # assert N % 2 == 0

        df = fs / N
        dt = df
        f = np.arange(0, N / 2 + 1) * df  # [0:df:f/2]

        dual_side = np.fft.fft(X)  # 双边谱
        half_idx = int(N / 2 + 1)
        single_side = dual_side[:, :, :, 0:half_idx]    # 传入形状和这个有关
        win = np.abs(single_side)

        ps = win ** 2
        if scaling == 'density':  # 计算功率谱密度
            scale = N * fs
        elif scaling == 'spectrum':  # 计算功率谱
            scale = N ** 2
        elif scaling is None:  # 不做缩放
            scale = 1
        else:
            raise ValueError('Unknown scaling: %r' % scaling)

        Pxy = ps / scale

        Pxy[:, 1:-1] *= 2  # 能量2倍;直流分量不用二倍, 中心频率点不用二倍

        return f, Pxy

