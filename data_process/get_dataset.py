import numpy as np

from utils.meta_info import dataset_class_dict


def get_dataset(args, step):
    data_load_path = f'{args.data_load_dir}/{args.dataset}/{args.data_id}/'

    if step == 'train':
        tr_x = np.load(data_load_path + 'tr_x.npy')
        tr_y = np.load(data_load_path + 'tr_y.npy')

        return dataset_class_dict[args.model](args, tr_x, tr_y)

    elif step == 'valid':
        vl_x = np.load(data_load_path + 'vl_x.npy')
        vl_y = np.load(data_load_path + 'vl_y.npy')

        return dataset_class_dict[args.model](args, vl_x, vl_y)

    elif step == 'test':
        ts_x = np.load(data_load_path + 'ts_x.npy')
        ts_y = np.load(data_load_path + 'ts_y.npy')

        return dataset_class_dict[args.model](args, ts_x, ts_y)
