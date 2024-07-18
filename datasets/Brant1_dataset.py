import os.path

import torch
from scipy import signal
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Brant1_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        self.seq_num, self.ch_num, seq_len, patch_len = x.shape

        self.x = x
        self.y = y

        # if os.path.exists(args.power_save_path):
        #     self.power = np.load(args.power_save_path)
        # else:
        #     self.power = self.compute_power(x, fs=256)
        #     np.save(args.power_save_path, self.power)
        self.power = self.compute_power(x, fs=256)

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        return self.x    [index, :, :, :], \
               self.power[index, :, :, :], \
               self.y    [index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)

    @staticmethod
    def compute_power(x, fs):
        f, Pxx_den = signal.periodogram(x, fs)

        f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
        poses = []
        for fi in range(len(f_thres) - 1):
            cond1_pos = np.where(f_thres[fi] < f)[0]
            cond2_pos = np.where(f_thres[fi + 1] >= f)[0]
            poses.append(np.intersect1d(cond1_pos, cond2_pos))

        ori_shape = Pxx_den.shape[:-1]
        Pxx_den = Pxx_den.reshape(-1, len(f))
        band_sum = [np.sum(Pxx_den[:, band_pos], axis=-1) + 1 for band_pos in poses]
        band_sum = [np.log10(_band_sum)[:, np.newaxis] for _band_sum in band_sum]
        band_sum = np.concatenate(band_sum, axis=-1)
        ori_shape += (8,)
        band_sum = band_sum.reshape(ori_shape)

        return band_sum

