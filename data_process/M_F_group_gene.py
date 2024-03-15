import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mayo_pat_ids =  [0, 18, 21, 1, 9, 19, 2, 5, 16, 3, 4, 23, 6, 7, 8, 14, 17, 20]
mayo_groups =   [[0, 18, 21], [1, 9, 19], [2, 5, 16], [3, 4, 23], [6, 7, 8], [14, 17, 20]]
fnusa_pat_ids = [1, 5, 2, 9, 3, 4, 12, 6, 7, 8, 10, 11, 13]
fnusa_groups =  [[1, 5], [2, 9], [3, 4, 12], [6, 7], [8, 10], [11, 13]]

class TrainArgs:
    max_epoch: int = 15
    warmup_epoch: int = 3

    train_batch_size: int = 256
    infer_batch_size: int = 256

    gpu_id: int = 0
    exp_id: int = 0

    last_epoch: int = -1
    load_his_ckpt: bool = True
    encoder_lr: float = 1e-4
    head_lr: float = 3e-4
    accu_step: int = 1

    epo_per_val: int = 1

    num_workers: int = 4

    patch_len: int = 1024
    seq_len: int = 16
    ff_head: int = 3

    group_num: int = 6
    class_num: int = 2
    hidden: bool = False
    normal_sample_num: int = 3000
    seizure_sample_num: int = 1000
    artifact_sample_num: int = 1000

    train_thres: int = 30
    dataset_name: str = None
    channel_num: int = 1
    baseline: bool = False

    data_root: str = '/data/brainnet/public_dataset'
    ckpt_path: str = '/data/yzz/Brant-2/model_ckpt/'
    data_save_dir: str = '/home/nas/share/TUEG/npy/mf'


def sample_data(data, label, normal_sample_num, seizure_sample_num, artifact_sample_num):
    normal_pos = np.where(label == 0)[0]
    seizure_pos = np.where(label == 1)[0]
    # artifact_pos = np.where(label == 2)[0]

    normal_pos = normal_pos[np.random.permutation(len(normal_pos))][:]
    seizure_pos = seizure_pos[np.random.permutation(len(seizure_pos))][:]
    # artifact_pos = artifact_pos[np.random.permutation(len(artifact_pos))][:artifact_sample_num]

    sample_pos = np.concatenate([normal_pos, seizure_pos], axis=0)
    return data[sample_pos], label[sample_pos]


def interpolate_data(args, dataset_name, g_id, interpolate_kind='cubic'):
    original_x = np.load(os.path.join(args.data_root, dataset_name, f'group_{g_id}_data.npy'))
    original_label = np.load(os.path.join(args.data_root, dataset_name, f'group_{g_id}_label.npy'))

    label_num = [np.sum(original_label == i) for i in range(3)]
    print(f'dataset {dataset_name}, group {g_id}: {label_num}')
    original_label[original_label == 2] = 0

    original_x, original_label = sample_data(original_x, original_label, args.normal_sample_num,
                                             args.seizure_sample_num, args.artifact_sample_num)
    seq_num, original_length = original_x.shape
    # new_length = args.patch_len * args.seq_len
    # interp_func = interp1d(np.arange(original_length), original_x, kind=interpolate_kind, axis=1)

    # new_x = interp_func(np.linspace(0, original_length - 1, new_length))
    # new_x = new_x.reshape(seq_num, args.seq_len, args.patch_len)
    new_x = original_x.reshape(seq_num, 15, 1000)   # 保持15000长度 不插值

    new_x = np.expand_dims(new_x, axis=1).astype(np.float32)
    np.save(os.path.join(args.data_root, dataset_name, f'group_data_15000_2to0/group_{g_id}_data.npy'), new_x)
    np.save(os.path.join(args.data_root, dataset_name, f'group_data_15000_2to0/group_{g_id}_label.npy'), original_label)
    print(f'interpolated data and label of dataset {dataset_name} group {g_id} saved')


def generate_pretrain_data(args, dataset_name):
    all_data = []
    for g_id in range(args.group_num):
        original_x = np.load(os.path.join(args.data_root, dataset_name, f'group_{g_id}_data.npy'))
        seq_num, original_length = original_x.shape
        new_length = args.patch_len * args.seq_len
        interp_func = interp1d(np.arange(original_length), original_x, kind='cubic', axis=1)

        new_x = interp_func(np.linspace(0, original_length - 1, new_length)).astype(np.float32)
        new_x = np.expand_dims(new_x, axis=0)
        all_data.append(new_x)

    all_data = np.concatenate(all_data, axis=1)
    np.save(os.path.join(args.data_save_dir, f'data_{dataset_name}.npy'), all_data)
    print(f'data of {dataset_name} saved')


if __name__ == '__main__':
    args = TrainArgs()
    for dataset_name in ['MAYO', 'FNUSA']:
        for g_id in range(args.group_num):
            interpolate_data(args, dataset_name, g_id)


