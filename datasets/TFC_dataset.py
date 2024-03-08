import numpy as np
from scipy import fft
from torch.utils.data import Dataset, DataLoader

from model.TFC.augmentation import DataTransform_TD, DataTransform_FD


class TFC_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        seq_num, ch_num, seq_len, patch_len = x.shape
        self.seq_num = seq_num
        self.ch_num = ch_num

        self.train_mode = args.train_mode
        self.n_class = len(np.unique(y))

        self.x = x
        self.y = y
        self.f = np.abs(fft.fft(self.x)) #/(window_length) # rfft for real value inputs.

        if args.train_mode == "unsupervised":  # no need to apply Augmentations in other modes
            self.aug1_x = DataTransform_TD(self.x, args)
            self.aug1_f = DataTransform_FD(self.f, args) # [7360, 1, 90]


    def __getitem__(self, index):
        if self.train_mode == "unsupervised":
            return self.x[index], self.aug1_x[index],  \
                   self.f[index], self.aug1_f[index]
        elif self.train_mode == 'finetune':
            return self.x[index], self.y[index], self.x[index], \
                   self.f[index], self.f[index]
        else:
            raise NotImplementedError(f'Undefined training mode {self.train_mode}')

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)
