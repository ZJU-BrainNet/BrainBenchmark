import random
import numpy as np
import time
import os
import sys
import argparse

from utils.misc import process_init
process_init()

from utils.meta_info import group_data_gene_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BrainBenchmark')

    group_database = parser.add_argument_group('DataGene')
    group_database.add_argument('--dataset', type=str, default='Clinical',  # MAYO FNUSA CHBMIT Siena Clinical SleepEDFx
                                help='What dataset we need to generate from.')
    group_database.add_argument('--sample_seq_num', type=float, default=None,
                                help='How many sequence samples in a group we need to sample from the dataset.')
    group_database.add_argument('--seq_len', type=float, default=15,
                                help='The number of patches in a sequence.')
    group_database.add_argument('--patch_len', type=float, default=100,
                                help='The number of points in a patch.')
    group_database.add_argument('--interpolate_kind', type=str, default='cubic',
                                help='What kind of interpolate when unifying the length.')
    group_database.add_argument('--data_save_dir', type=str, default='/data/brainnet/benchmark/gene_data/',
                                help='The path to save the generated data.')

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
        args.dataset,
        args.sample_seq_num,
        args.seq_len,
        args.patch_len,
    )

    group_data_gene_dict[args.dataset](args)





