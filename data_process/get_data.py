import pickle

import numpy as np

from data_process.data_info import data_info_dict


def default_get_data(args, step,):
    data_load_path = f'{args.data_load_dir}/{args.data_id}/'

    group_num      = data_info_dict[args.dataset]['group_num']
    split          = data_info_dict[args.dataset]['split']
    various_ch_num = data_info_dict[args.dataset]['various_ch_num']

    indices = list(range(group_num))
    shift = args.cv_id
    indices = indices[-shift:] + indices[:-shift]

    tr_indices = indices[ : split[0]]
    vl_indices = indices[split[0] : -split[2]]
    ts_indices = indices[-split[2] : ]

    if step == 'train':
        target_indices = tr_indices
    elif step == 'valid':
        target_indices = vl_indices
    elif step == 'test':
        target_indices = ts_indices
    else:
        raise NotImplementedError('Unknown step.')

    print(f'{step} group indices: {target_indices}')

    group_x_list, group_y_list = [], []
    for g_id in target_indices:
        x = np.load(data_load_path + f'group{g_id}_x.npy')
        y = np.load(data_load_path + f'group{g_id}_y.npy')
        group_x_list.append(x)
        group_y_list.append(y)

    if not various_ch_num:
        group_x_list = [np.concatenate(group_x_list, axis=0)]
        group_y_list = [np.concatenate(group_y_list, axis=0)]

    return group_x_list, group_y_list



def clinical_get_data(args, step):
    data_load_path = f'{args.data_load_dir}/{args.data_id}/'

    group_num      = data_info_dict[args.dataset]['group_num']
    split          = data_info_dict[args.dataset]['split']
    various_ch_num = data_info_dict[args.dataset]['various_ch_num']

    indices = list(range(1, group_num+1))   # g1, g2, g3, g4
    shift = args.cv_id
    indices = indices[-shift:] + indices[:-shift]

    tr_indices = indices[ : split[0]]
    vl_indices = indices[split[0] : -split[2]]
    ts_indices = indices[-split[2] : ]

    if step == 'train':
        target_indices = tr_indices
    elif step == 'valid':
        target_indices = vl_indices
    elif step == 'test':
        target_indices = ts_indices
    else:
        raise NotImplementedError('Unknown step.')

    print(f'{step} group indices: {target_indices}')

    group_x_list, group_y_list = [], []
    for g_id in target_indices:
        if step != 'test':
            x = pickle.load(open(data_load_path + f'sampled_g{g_id}_x.pkl', 'rb'))
            y = pickle.load(open(data_load_path + f'sampled_g{g_id}_y.pkl', 'rb'))
        else:
            x = pickle.load(open(data_load_path + f'unsampled_g{g_id}_x.pkl', 'rb'))
            y = pickle.load(open(data_load_path + f'unsampled_g{g_id}_y.pkl', 'rb'))
        group_x_list += x
        group_y_list += y

    if not various_ch_num:
        group_x_list = [np.concatenate(group_x_list, axis=0)]
        group_y_list = [np.concatenate(group_y_list, axis=0)]

    return group_x_list, group_y_list


if __name__ == '__main__':
    indices = list(range(1, 5))  # g1, g2, g3, g4
    shift = 1
    indices = indices[-shift:] + indices[:-shift]

    indices = list(range(1, 5))   # g1, g2, g3, g4
    for shift in range(5):
        indices = indices[-shift:] + indices[:-shift]

        tr_indices = indices[ : split[0]]
        vl_indices = indices[split[0] : -split[2]]
        ts_indices = indices[-split[2] : ]
