import random
import numpy as np
import time
import os
import sys
import argparse

from utils.meta_info import dataset_dict, dataset_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BrainBenchmark')

    group_database = parser.add_argument_group('DataGene')
    group_database.add_argument('--data_save_dir', type=str, default='/data/brainnet/benchmark/gene_data/',
                                help='The path to save the generated data.')
    group_database.add_argument('--model', type=str, default='Brant',
                                help='What model these generated data is for.')
    group_database.add_argument('--sample_seq_num', type=float, default=2400,
                                help='How many sequence samples we need to sample from each dataset.')
    group_database.add_argument('--seq_len', type=float, default=16,
                                help='The number of patches in a sequence.')
    group_database.add_argument('--patch_len', type=float, default=500,
                                help='The number of points in a patch.')
    group_database.add_argument('--interpolate_kind', type=str, default='cubic',
                                help='What kind of interpolate when unifying the length.')

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    for dataset_name in dataset_dict.keys():
        args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
            dataset_name,
            args.sample_seq_num,
            args.seq_len,
            args.patch_len,
        )
        dataset_dict[dataset_name](args)





