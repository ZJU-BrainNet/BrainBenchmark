import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DefaultDataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        self.seq_num, self.ch_num, seq_len, patch_len = x.shape

        self.x = x
        self.y = y

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)


    def __getitem__(self, index):
        return self.x[index, :, :, :], \
               self.y[index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)
