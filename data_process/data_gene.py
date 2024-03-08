import random

import numpy as np
import os
from scipy.interpolate import interp1d

from utils.misc import make_dir_if_not_exist


def sample_and_unify_length(args, sample_seq_num, x, y, ):
    seq_num, ori_len = x.shape

    # sampling some samples
    indices = list(range(seq_num))
    random.shuffle(indices)
    indices = indices[0 : sample_seq_num]
    x = x[indices, :]
    y = y[indices, ]

    # unify the length
    new_len = args.patch_len * args.seq_len

    interp_func = interp1d(np.arange(ori_len), x, kind=args.interpolate_kind, axis=1)
    new_x = interp_func(np.linspace(0, ori_len - 1, new_len))
    new_x = new_x.reshape(sample_seq_num, args.seq_len, args.patch_len)    # (args.sample_seq_num, seq_len, patch_len)

    new_x = np.expand_dims(new_x, axis=1).astype(np.float32)     # (args.sample_seq_num, ch_num=1, seq_len, patch_len)
    return new_x, y


def DataGene_MAYO(args):
    data_path = '/data/brainnet/public_dataset/MAYO/'

    x_group_list, y_group_list = [], []
    for i in range(0, 6):
        x = np.load(data_path + f'group_{i}_data.npy')
        y = np.load(data_path + f'group_{i}_label.npy')

        x_group_list.append(x)
        y_group_list.append(y)

    indices = list(range(0, 6))
    random.shuffle(indices)

    tr_indices = indices[0:4]   # 3
    vl_indices = indices[4:5]   # 1
    ts_indices = indices[5:6]   # 2

    tr_x = np.concatenate([x_group_list[i] for i in tr_indices], axis=0)
    vl_x = np.concatenate([x_group_list[i] for i in vl_indices], axis=0)
    ts_x = np.concatenate([x_group_list[i] for i in ts_indices], axis=0)
    tr_y = np.concatenate([y_group_list[i] for i in tr_indices], axis=0)
    vl_y = np.concatenate([y_group_list[i] for i in vl_indices], axis=0)
    ts_y = np.concatenate([y_group_list[i] for i in ts_indices], axis=0)

    tr_x, tr_y = sample_and_unify_length(args, args.sample_seq_num // 6 * 3, tr_x, tr_y)
    vl_x, vl_y = sample_and_unify_length(args, args.sample_seq_num // 6 * 1, vl_x, vl_y)
    ts_x, ts_y = sample_and_unify_length(args, args.sample_seq_num // 6 * 2, ts_x, ts_y)

    # label: 0,1,2 => 0,1
    tr_y[np.where(tr_y == 2)] = 0
    vl_y[np.where(vl_y == 2)] = 0
    ts_y[np.where(ts_y == 2)] = 0

    save_path = f'{args.data_save_dir}/MAYO/{args.data_id}/'
    make_dir_if_not_exist(save_path)
    np.save(save_path + 'tr_x', tr_x)
    np.save(save_path + 'vl_x', vl_x)
    np.save(save_path + 'ts_x', ts_x)
    np.save(save_path + 'tr_y', tr_y)
    np.save(save_path + 'vl_y', vl_y)
    np.save(save_path + 'ts_y', ts_y)
    print(f'Generated data of MAYO saved in {save_path}.')


def DataGene_FNUSA(args):
    data_path = '/data/brainnet/public_dataset/FNUSA/'

    x_group_list, y_group_list = [], []
    for i in range(0, 6):
        x = np.load(data_path + f'group_{i}_data.npy')
        y = np.load(data_path + f'group_{i}_label.npy')

        x_group_list.append(x)
        y_group_list.append(y)

    indices = list(range(0, 6))
    random.shuffle(indices)

    tr_indices = indices[0:4]   # 3
    vl_indices = indices[4:5]   # 1
    ts_indices = indices[5:6]   # 2

    tr_x = np.concatenate([x_group_list[i] for i in tr_indices], axis=0)
    vl_x = np.concatenate([x_group_list[i] for i in vl_indices], axis=0)
    ts_x = np.concatenate([x_group_list[i] for i in ts_indices], axis=0)
    tr_y = np.concatenate([y_group_list[i] for i in tr_indices], axis=0)
    vl_y = np.concatenate([y_group_list[i] for i in vl_indices], axis=0)
    ts_y = np.concatenate([y_group_list[i] for i in ts_indices], axis=0)

    tr_x, tr_y = sample_and_unify_length(args, args.sample_seq_num // 6 * 3, tr_x, tr_y)
    vl_x, vl_y = sample_and_unify_length(args, args.sample_seq_num // 6 * 1, vl_x, vl_y)
    ts_x, ts_y = sample_and_unify_length(args, args.sample_seq_num // 6 * 2, ts_x, ts_y)

    # label: 0,1,2 => 0,1
    tr_y[np.where(tr_y == 2)] = 0
    vl_y[np.where(vl_y == 2)] = 0
    ts_y[np.where(ts_y == 2)] = 0

    save_path = f'{args.data_save_dir}/FNUSA/{args.data_id}/'
    make_dir_if_not_exist(save_path)
    np.save(save_path + 'tr_x', tr_x)
    np.save(save_path + 'vl_x', vl_x)
    np.save(save_path + 'ts_x', ts_x)
    np.save(save_path + 'tr_y', tr_y)
    np.save(save_path + 'vl_y', vl_y)
    np.save(save_path + 'ts_y', ts_y)
    print(f'Generated data of FNUSA saved in {save_path}.')