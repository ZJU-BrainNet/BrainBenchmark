import numpy as np
import torch
from scipy import fft
from torch.utils.data import Dataset, DataLoader

from model.TFC.augmentation import DataTransform_TD, DataTransform_FD


class TFC_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        self.seq_num, self.ch_num, seq_len, patch_len = x.shape

        self.run_mode = args.run_mode

        self.x = x
        self.y = y
        self.f = np.abs(fft.fft(self.x)) #/(window_length) # rfft for real value inputs.

        if args.run_mode == "unsupervised":  # no need to apply Augmentations in other modes
            self.aug1_x = DataTransform_TD(self.x, args)
            self.aug1_f = DataTransform_FD(self.f, args) # [7360, 1, 90]

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        if self.run_mode == "unsupervised":
            return self.x[index], self.aug1_x[index],  \
                   self.f[index], self.aug1_f[index]
        elif self.run_mode == 'finetune' or self.run_mode == 'test':
            return self.x[index], self.y[index], self.x[index], \
                   self.f[index], self.f[index]
        else:
            raise NotImplementedError(f'Undefined running mode {self.run_mode}')

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)
