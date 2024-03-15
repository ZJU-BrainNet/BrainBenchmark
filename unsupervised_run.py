import random
import shutil

import torch
import numpy as np
import time
import os
import sys
import argparse
import json
from copy import deepcopy

from utils.misc import set_seed, load_checkpoint, save_logs, save_checkpoint, update_main_logs, show_logs, process_init
process_init()

from model.ch_aggr_clsf import ChannelAggrClsf
from pipeline.eval_epoch import evaluate_epoch
from pipeline.train_epoch import train_epoch
from utils.meta_info import model_dict, get_data_dict, trainer_dict
from data_process.data_info import data_info_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BrainBenchmark')

    group_train = parser.add_argument_group('Train')
    group_train.add_argument('--exp_id', type=str, default='-1',
                             help='The experimental id.')
    group_train.add_argument('--gpu_id', type=int, default=1,
                             help='The gpu id.')
    group_train.add_argument('--cv_id', type=int, default=0,
                             help='The cross validation id.')
    group_train.add_argument('--run_mode', type=str, default='test',   # unsupervised, finetune, test
                             help='To perform self-/unsupervised training, finetuning, or testing.')
    group_train.add_argument('--batch_size', type=int, default=128,
                             help='Number of batches.')
    group_train.add_argument('--save_epochs', type=int, default=3,
                             help='The epoch number to save checkpoint.')
    group_train.add_argument('--epoch_num', type=int, default=100,
                             help='Epoch number for total iterations.')
    group_train.add_argument('--early_stop', action='store_false',
                             help='Whether to use early stopping technique during training.')
    group_train.add_argument('--patience', type=int, default=6,
                             help='The waiting epoch number for early stopping.')
    group_train.add_argument('--load_ckpt_path', type=str, default='/data/brainnet/benchmark/ckpt/',    # None '/data/brainnet/benchmark/ckpt/'
                             help='The path to load checkpoint (.pt file or the upper path).')
    group_train.add_argument('--load_best', action='store_false',
                             help='Whether to load the best state in the checkpoints (to continue unsupervised training or begin finetuning).')
    group_train.add_argument('--save_ckpt_path', type=str, default='/data/brainnet/benchmark/ckpt/',
                             help='The path to save checkpoint')
    group_train.add_argument('--lr', type=float, default=3e-4,
                             help='The learning rate.')
    group_train.add_argument('--gpu', action='store_false',
                             help='Whether to load the data and model to GPU.')
    group_train.add_argument('--tqdm_dis', action='store_true', # add --tqdm_dis to disable
                             help='Whether to use tqdm_dis')

    group_data = parser.add_argument_group('Data')
    group_data.add_argument('--dataset', type=str, default='FNUSA',  # MAYO FNUSA CHBMIT Siena Clinical SleepEDFx
                            help='The dataset to perform training.')
    group_data.add_argument('--sample_seq_num', type=float, default=None,
                            help='The number of sequence sampled from each dataset.')
    group_data.add_argument('--seq_len', type=float, default=15,
                            help='The number of patches in a sequence.')
    group_data.add_argument('--patch_len', type=float, default=100,
                            help='The number of points in a patch.')
    group_data.add_argument('--data_load_dir', type=str, default='/data/brainnet/benchmark/gene_data/',
                            help='The path to load the generated data.')
    group_data.add_argument('--n_process_loader', type=int, default=10,
                            help='Number of processes to call to load the dataset.')

    group_arch = parser.add_argument_group('Architecture')
    group_arch.add_argument('--random_seed', type=int, default=None,
                            help="Set a specific random seed.")
    group_arch.add_argument('--model', type=str, default='TFC',
                            help='The model to run.')
    group_arch.add_argument('--cnn_in_channels', type=int, default=10,
                            help="The number of input channels of the dataset.")
    group_arch.add_argument('--cnn_kernel_size', type=int, default=8,
                            help="The kernel size of the CNN to aggregate the channels.")
    group_arch.add_argument('--final_dim', type=int, default=512,
                            help="The dim of final representations.")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    trainer = trainer_dict[args.model](args)

    args = trainer.set_config(parser)
    args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
        args.dataset,
        args.sample_seq_num,
        args.seq_len,
        args.patch_len,
    )

    if args.random_seed is None:
        args.random_seed = random.randint(0, 2 ** 31)
    set_seed(args.random_seed)

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=False)}')
    print('-' * 50)

    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    main_logs = {"epoch": []}

    if args.run_mode != 'test':
        tr_x_list, tr_y_list = get_data_dict[args.dataset](args, step='train')
        vl_x_list, vl_y_list = get_data_dict[args.dataset](args, step='valid')

    args.n_class = data_info_dict[args.dataset]['n_class']

    model = model_dict[args.model](args).to(device)
    clsf  = ChannelAggrClsf(args).to(device)

    loss_func = trainer.clsf_loss_func(args)

    optimizer = torch.optim.AdamW([
         {'params': list(model.parameters()), 'lr': args.lr},
         {'params': list(clsf .parameters()), 'lr': args.lr},
         ],
        betas=(0.9, 0.95), eps=1e-5,
    )


    best_vl_loss = np.inf
    state_dict = None

    # host_name = os.getcwd().split('/')[2]
    # args.path_checkpoint = utils.paths[host_name]  # 根据用户设置path_checkpoint

    # if run_mode == unsupervised, will load the unsupervised ckpt to continue from break point
    # if run_mode == finetune, will also load the unsupervised ckpt to finetune
    # if run_mode == test, will load the finetune ckpt to infer
    if args.run_mode == 'unsupervised' or args.run_mode == 'finetune':
        ckpt_type = 'unsupervised'
    elif args.run_mode == 'test':
        ckpt_type = 'finetune'
    else:
        raise NotImplementedError(f'Unknown run_mode {args.run_mode}.')

    # Load ckpt
    if args.load_ckpt_path is not None:
        args.load_ckpt_path = f'{args.load_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/'

        load_path, main_logs = load_checkpoint(args, ckpt_type)
        print('Load checkpoint:', load_path)

        state_dict = torch.load(load_path, 'cpu')
        if args.load_best or args.run_mode == 'test':
            model.load_state_dict(state_dict["BestModel"], strict=False)
            clsf .load_state_dict(state_dict["BestClsf"], strict=False)
        else:
            model.load_state_dict(state_dict["Model"], strict=False)
            clsf .load_state_dict(state_dict["Clsf"], strict=False)

        if args.run_mode == 'unsupervised':
            best_vl_loss = state_dict["BestValLoss"]
            best_model_state = state_dict["BestModel"]
            best_clsf_state  = state_dict["BestClsf"]
            optimizer.load_state_dict(state_dict["Optimizer"])
            print(f'Checkpoint loaded: best_vl_loss = {best_vl_loss:.4f} Now continue the unsupervised training.')
    else:
        print('Not load checkpoint')
        best_model_state = deepcopy(model.state_dict())
        best_clsf_state  = deepcopy(clsf .state_dict())

    if args.run_mode == 'test':
        if args.load_ckpt_path is None:
            print(f'WARNING: You are inferring with an initialized model. Please do not set the args.load_ckpt_path None.')
        ts_x_list, ts_y_list = get_data_dict[args.dataset](args, step='test')
        ts_logs, ts_loss = evaluate_epoch(args, ts_x_list, ts_y_list, model, clsf, loss_func, step='test')
        show_logs('[Test]', ts_logs, None)
        exit(0)

    # Settings to save ckpt
    if args.save_ckpt_path is not None:
        args.save_ckpt_path = f'{args.save_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/{args.run_mode}_ckpt/'
        # create if not exist
        if not os.path.exists(args.save_ckpt_path):
            os.makedirs(args.save_ckpt_path)

    print('-' * 50)

    print(f"Running at most {args.epoch_num} epochs")
    start_epoch = len(main_logs["epoch"]) if args.run_mode == 'unsupervised' else 0   # continue when restart
    last_tr_loss = 0
    wait_epoch = 0

    start_time = time.time()
    for epoch in range(start_epoch, args.epoch_num):
        print('-' * 50)
        print(f"Epoch {epoch}")
        # cpu_stats()

        tr_logs, tr_loss = train_epoch(args, tr_x_list, tr_y_list, model, clsf, loss_func, optimizer, )
        vl_logs, vl_loss = evaluate_epoch(args, vl_x_list, vl_y_list, model, clsf, loss_func, step='valid')

        # print(f'Ran {epoch - start_epoch + 1} epochs in {time.time() - start_time:.2f} seconds')

        # process validation loss
        if vl_loss < best_vl_loss:
            print(f'Best model state updated. Best valid loss {best_vl_loss:.4f} => {vl_loss:.4f}')
            best_vl_loss = deepcopy(vl_loss)
            best_model_state = deepcopy(model.state_dict())
            best_clsf_state  = deepcopy(clsf .state_dict())
            wait_epoch = 0
        elif args.early_stop:
            wait_epoch += 1
            print(f'Early stop: wait epoch {wait_epoch}/{args.patience}')

        # update main logs
        main_logs = update_main_logs(main_logs, tr_logs, vl_logs, epoch)

        # save checkpoint at every epoch
        if args.save_ckpt_path is not None and (
                epoch % args.save_epochs == 0 or epoch == args.epoch_num - 1 or
                wait_epoch >= args.patience):
            model_state = model.state_dict()
            clsf_state  = clsf .state_dict()
            optimizer_state = optimizer.state_dict()

            save_checkpoint(
                model_state,
                clsf_state,
                optimizer_state,
                best_model_state,
                best_clsf_state,
                best_vl_loss,
                f"{args.save_ckpt_path}/{epoch}.pt",
            )
            save_logs(main_logs, f"{args.save_ckpt_path}/logs.json")
            print('Checkpoint and main logs saved.')

        if wait_epoch >= args.patience:
            break

    print('-' * 10, 'Training finished', '-' * 10)


