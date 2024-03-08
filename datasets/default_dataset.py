import numpy as np
from torch.utils.data import Dataset, DataLoader


class DefaultDataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        seq_num, ch_num, seq_len, patch_len = x.shape
        self.seq_num = seq_num

        self.x = x
        self.y = y

        self.n_class = len(np.unique(y))


    def __getitem__(self, index):
        return self.x[index, :, :, :], \
               self.y[index, ]


    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)
