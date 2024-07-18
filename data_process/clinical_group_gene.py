import glob
import os
import pickle

import numpy as np

from data_process.data_info import data_info_dict
from data_process.defult_group_gene import sample_and_unify_length
from utils.misc import make_dir_if_not_exist

clinical_group_dict = {
    'g1': ['01TGX', '09WYT'],
    'g2': ['07WF', '08CXF'],
    'g3': ['03ZXY', '06ZYJ'],
    'g4': ['02GJX', '05ZLH'],
}

# exp_dict = {
#     1:  [clinical_group_dict['g3'] + clinical_group_dict['g4'], clinical_group_dict['g2'], clinical_group_dict['g1']],
#     2:  [clinical_group_dict['g2'] + clinical_group_dict['g4'], clinical_group_dict['g3'], clinical_group_dict['g1']],
#     3:  [clinical_group_dict['g2'] + clinical_group_dict['g3'], clinical_group_dict['g4'], clinical_group_dict['g1']],
#     4:  [clinical_group_dict['g3'] + clinical_group_dict['g4'], clinical_group_dict['g1'], clinical_group_dict['g2']],
#     5:  [clinical_group_dict['g1'] + clinical_group_dict['g4'], clinical_group_dict['g3'], clinical_group_dict['g2']],
#     6:  [clinical_group_dict['g1'] + clinical_group_dict['g3'], clinical_group_dict['g4'], clinical_group_dict['g2']],
#     7:  [clinical_group_dict['g2'] + clinical_group_dict['g4'], clinical_group_dict['g1'], clinical_group_dict['g3']],
#     8:  [clinical_group_dict['g1'] + clinical_group_dict['g4'], clinical_group_dict['g2'], clinical_group_dict['g3']],
#     9:  [clinical_group_dict['g1'] + clinical_group_dict['g2'], clinical_group_dict['g4'], clinical_group_dict['g3']],
#     10: [clinical_group_dict['g2'] + clinical_group_dict['g3'], clinical_group_dict['g1'], clinical_group_dict['g4']],
#     11: [clinical_group_dict['g1'] + clinical_group_dict['g3'], clinical_group_dict['g2'], clinical_group_dict['g4']],
#     12: [clinical_group_dict['g1'] + clinical_group_dict['g2'], clinical_group_dict['g3'], clinical_group_dict['g4']],
# }



def sampled_group_data_gene(args):
    file_dict = {
        '01TGX': 't01_14ki',
        '02GJX': 't02_127w',
        '03ZXY': 't03_1461',
        '05ZLH': 't05_10ap',
        '06ZYJ': 't06_11mb',
        '07WF' : 't07_123z',
        '08CXF': 't08_1295',
        '09WYT': 't09_16j3',
    }
    data_path = data_info_dict[args.dataset]['sampled_data_path']

    for group in clinical_group_dict.keys():
        group_x, group_y = [], []
        for subj in clinical_group_dict[group]:
            x = np.load(f'{data_path}/{subj}/{file_dict[subj]}/data.npy')
            y = np.load(f'{data_path}/{subj}/{file_dict[subj]}/label.npy')
            # x: (ch_num, broad_num, 15, 1500)
            # y: (ch_num, broad_num, 15)
            ch_num, broad_num, _, _ = x.shape

            if args.seq_len == 15 and args.patch_len == 100:
                x = np.reshape(x, (ch_num, broad_num*15, 15*100))
                y = np.reshape(y, (ch_num, broad_num*15))

            elif args.seq_len == 15 and args.patch_len == 1500:
                x = np.reshape(x, (ch_num, broad_num, 15*1500))  # x: (ch_num, broad_num, 15*1500)
                y = np.mean(y, axis=-1)                                   # y: (ch_num, broad_num,)

            x = np.swapaxes(x, axis1=0, axis2=1)    # (seq_num, ch_num, seq_len)
            y = np.swapaxes(y, axis1=0, axis2=1)    # (seq_num, ch_num,)

            x, y = sample_and_unify_length(args, args.sample_seq_num, x, y,)

            group_x.append(x)
            group_y.append(y.astype(np.int64))

        save_path = f'{args.data_save_dir}/{args.data_id}/'
        make_dir_if_not_exist(save_path)
        pickle.dump(group_x, open(f'{save_path}/sampled_{group}_x.pkl','wb'))
        pickle.dump(group_y, open(f'{save_path}/sampled_{group}_y.pkl','wb'))
        print(f'Generated sampled {group} data of {args.dataset} saved in {save_path}.')


def unsampled_group_data_gene(args):
    file_dict = {
        '01TGX': 'FA0014KI',
        '02GJX': 'FA00127W',
        '03ZXY': 'FA001461',
        '05ZLH': 'FA0010AP',
        '06ZYJ': 'FA0011MB',
        '07WF': 'FA00123Z',
        '08CXF': 'FA001295',
        '09WYT': 'FA0016J3',
    }
    def remove_suffix(input_string, suffixes=('_data.npy', '_label.npy')):
        for suffix in suffixes:
            if input_string.endswith(suffix):
                # 使用 rsplit 分割字符串，限制分割次数为1次，即只分割最后一个后缀
                return input_string.rsplit(suffix, 1)[0]
        return input_string

    data_path = data_info_dict[args.dataset]['unsampled_data_path']

    for group in clinical_group_dict.keys():
        group_x, group_y = [], []
        for subj in clinical_group_dict[group]:
            files = glob.glob(f'{data_path}/{subj}/{file_dict[subj]}/{file_dict[subj]}-*')
            files = [remove_suffix(file) for file in files]
            files = np.unique(files)
            if len(files) == 0:
                # read the file without -0,-1,...
                files = [f'{data_path}/{subj}/{file_dict[subj]}/{file_dict[subj]}']

            for file in files:
                print(f'Dealing with file: {file}')

                x = np.load(f'{file}_data.npy')
                y = np.load(f'{file}_label.npy')
                fs = pickle.load(open(f'{data_path}/{subj}/sample_rate_dict.pkl', 'rb'))
                fs = fs[file_dict[subj]]

                # downsample to 250Hz
                k = fs // 250
                x = x[:, ::k]
                y = y[:, ::k]

                # handle the illegal label
                y[np.where(y>=2)] = 0
                y[np.where(y<0)] = 2    # when calculating metrics, label=2 will be omitted

                # normalization
                new_x = []
                for ch_idx in range(x.shape[0]):
                    arr = x[ch_idx, :]
                    normalized_arr = (arr - arr.mean()) / arr.std()
                    new_x.append(normalized_arr)
                x = np.stack(new_x, axis=0)

                # x: (ch_num, tot_len)
                # y: (ch_num, tot_len)
                ch_num, tot_len = x.shape
                target_len = args.seq_len * args.patch_len
                x = x[:, : tot_len // target_len * target_len]
                y = y[:, : tot_len // target_len * target_len]
                x = x.reshape(ch_num, -1, target_len)
                y = y.reshape(ch_num, -1, target_len)
                x = np.swapaxes(x, axis1=0, axis2=1)    # (seq_num, ch_num, args.seq_len * args.patch_len)
                y = np.swapaxes(y, axis1=0, axis2=1)    # (seq_num, ch_num, args.seq_len * args.patch_len)

                y = np.max(y, axis=2)

                x, y = sample_and_unify_length(args, args.sample_seq_num, x, y,)

                group_x.append(x)
                group_y.append(y.astype(np.int64))

        save_path = f'{args.data_save_dir}/{args.data_id}/'
        make_dir_if_not_exist(save_path)
        pickle.dump(group_x, open(f'{save_path}/unsampled_{group}_x.pkl','wb'))
        pickle.dump(group_y, open(f'{save_path}/unsampled_{group}_y.pkl','wb'))
        print(f'Generated unsampled {group} data of {args.dataset} saved in {save_path}.')


def clinical_group_data_gene(args):
    # sampled_group_data_gene(args)
    unsampled_group_data_gene(args)
