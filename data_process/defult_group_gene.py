import random

import numpy as np
import os
from scipy.interpolate import interp1d

from data_process.data_info import data_info_dict
from utils.misc import make_dir_if_not_exist


def sample_and_unify_length(args, sample_seq_num, x, y, ):
    seq_num, ch_num, ori_len = x.shape

    if sample_seq_num is not None:    # if None, use all the seq samples
        # sampling some samples
        indices = list(range(seq_num))
        random.shuffle(indices)
        indices = indices[0 : sample_seq_num]
        x = x[indices, :]
        y = y[indices, ]

    # unify the length
    new_len = args.patch_len * args.seq_len

    if ori_len == new_len:
        x = x.reshape(-1, ch_num, args.seq_len, args.patch_len)
        return x, y

    x = x.reshape(-1, ori_len) # (seq_num or sample_seq_num * ch_num, ori_len)

    if ori_len % new_len == 0:
        k = ori_len // new_len
        new_x = x[:, ::k]
    else:
        interp_func = interp1d(np.arange(ori_len), x, kind=args.interpolate_kind, axis=1)
        new_x = interp_func(np.linspace(0, ori_len - 1, new_len))
    new_x = new_x.reshape(-1, ch_num, args.seq_len, args.patch_len)    # (seq_num or sample_seq_num, ch_num, seq_len, patch_len)
    return new_x, y


def default_group_data_gene(args):
    data_path = data_info_dict[args.dataset]['data_path']
    group_num = data_info_dict[args.dataset]['group_num']

    for i in range(0, group_num):
        x = np.load(data_path + f'group_{i}_data.npy')
        y = np.load(data_path + f'group_{i}_label.npy')

        _, ch_num, seq_len, patch_len = x.shape
        x = x.reshape(-1, ch_num, seq_len * patch_len)

        x, y = sample_and_unify_length(args, args.sample_seq_num, x, y)

        save_path = f'{args.data_save_dir}/{args.data_id}/'
        make_dir_if_not_exist(save_path)
        np.save(save_path + f'group{i}_x.npy', x)
        np.save(save_path + f'group{i}_y.npy', y)
        print(f'Generated group{i} data of {args.dataset} saved in {save_path}.')
