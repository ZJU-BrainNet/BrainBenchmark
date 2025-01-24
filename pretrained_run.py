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
from torch import nn

from utils.misc import set_seed, load_checkpoint, save_logs, save_checkpoint, update_main_logs, show_logs, process_init
process_init()

from model.ch_aggr_clsf import ChannelAggrClsf
from pipeline.eval_epoch import evaluate_epoch
from pipeline.train_epoch import train_epoch
from utils.meta_info import get_data_dict, \
    trainer_dict, model_dict
from data_process.data_info import data_info_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BrainBenchmark')

    group_train = parser.add_argument_group('Train')
    group_train.add_argument('--exp_id', type=str, default='-1',
                             help='The experimental id.')
    group_train.add_argument('--gpu_id', type=int, default=5,
                             help='The gpu id.')
    group_train.add_argument('--cv_id', type=int, default=0,
                             help='The cross validation id.')
    group_train.add_argument('--run_mode', type=str, default='finetune',   # finetune, test
                             help='To perform finetuning, or testing.')
    group_train.add_argument('--batch_size', type=int, default=8,
                             help='Number of batches.')
    group_train.add_argument('--save_epochs', type=int, default=5,
                             help='The epoch number to save checkpoint.')
    group_train.add_argument('--epoch_num', type=int, default=15,
                             help='Epoch number for total iterations.')
    group_train.add_argument('--early_stop', action='store_false',
                             help='Whether to use early stopping technique during training.')
    group_train.add_argument('--patience', type=int, default=10,
                             help='The waiting epoch number for early stopping.')
    group_train.add_argument('--from_pretrained', action='store_false', # when not --from_pretrained, must set the load_ckpt_path
                             help='Whether to finetune from pretrained weights')
    group_train.add_argument('--load_ckpt_path', type=str, default='/data/brainnet/benchmark/ckpt/',    # None '/data/brainnet/benchmark/ckpt/'
                             help='The path to load checkpoint (.pt file or the upper path).')
    group_train.add_argument('--load_best', action='store_false',
                             help='Whether to load the best state in the checkpoints (to continue unsupervised training or begin finetuning).')
    group_train.add_argument('--save_ckpt_path', type=str, default='/data/brainnet/benchmark/ckpt/',
                             help='The path to save checkpoint')
    group_train.add_argument('--model_lr', type=float, default=1e-5,
                             help='The learning rate of the pretrained model.')
    group_train.add_argument('--clsf_lr', type=float, default=3e-4,
                             help='The learning rate of the classifier.')
    group_train.add_argument('--tqdm_dis', action='store_true', # add --tqdm_dis to disable
                             help='Whether to use tqdm_dis')

    group_data = parser.add_argument_group('Data')
    group_data.add_argument('--dataset', type=str, default='Clinical',  # MAYO FNUSA CHBMIT Siena Clinical SleepEDFx SeizureA SeizureC SeizureB UCSD_ON UCSD_OFF HUSM RepOD
                            help='The dataset to perform training.')
    group_data.add_argument('--sample_seq_num', type=int, default=None,
                            help='The number of sequence sampled from each dataset.')
    group_data.add_argument('--seq_len', type=int, default=15,
                            help='The number of patches in a sequence.')
    group_data.add_argument('--patch_len', type=int, default=100,
                            help='The number of points in a patch.')
    group_data.add_argument('--data_load_dir', type=str, default='/data/brainnet/benchmark/gene_data/',
                            help='The path to load the generated data.')
    group_data.add_argument('--n_process_loader', type=int, default=5,
                            help='Number of processes to call to load the dataset.')

    group_arch = parser.add_argument_group('Architecture')
    group_arch.add_argument('--random_seed', type=int, default=None,
                            help="Set a specific random seed.")
    group_arch.add_argument('--model', type=str, default='Brant1',   # BrainBERT GPT4TS Brant1 Brant2 BIOT LaBraM
                            help='The model to run.')
    group_arch.add_argument('--cnn_in_channels', type=int, default=10,
                            help="The number of input channels of the dataset.")
    group_arch.add_argument('--cnn_kernel_size', type=int, default=8,
                            help="The kernel size of the CNN to aggregate the channels.")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    trainer = trainer_dict[args.model](args)

    args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
        args.dataset,
        args.sample_seq_num,
        args.seq_len,
        args.patch_len,
    )
    args = trainer.set_config(args)

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
    clsf = ChannelAggrClsf(args).to(device)


    loss_func = trainer.clsf_loss_func(args)
    optimizer = trainer.optimizer(args, model, clsf)
    scheduler = trainer.scheduler(optimizer)


    best_vl_loss = np.inf
    state_dict = None

    # if run_mode == finetune and from_pretrained == True, will load the pretrained weights
    # if run_mode == finetune and from_pretrained == False, will load the ckpt to continue finetuning
    # if run_mode == test, will load the finetuned ckpt to infer

    assert (not args.from_pretrained and args.load_ckpt_path is not None) or \
           args.from_pretrained or \
           args.run_mode == 'test'      # must set the load_ckpt_path if continue finetuning

    # host_name = os.getcwd().split('/')[2]
    # args.path_checkpoint = utils.paths[host_name]  # 根据用户设置path_checkpoint

    # Load ckpt to continue finetuning
    if (not args.from_pretrained and args.load_ckpt_path is not None) or \
       args.run_mode == 'test':
        args.load_ckpt_path = f'{args.load_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/'

        load_path, main_logs = load_checkpoint(args, ckpt_type='finetune')
        print('Load checkpoint:', load_path)

        state_dict = torch.load(load_path, 'cpu')
        if args.load_best or args.run_mode == 'test':
            model.load_state_dict(state_dict["BestModel"], strict=False)
            clsf .load_state_dict(state_dict["BestClsf"], strict=False)
        else:
            model.load_state_dict(state_dict["Model"], strict=False)
            clsf .load_state_dict(state_dict["Clsf"], strict=False)

        best_vl_loss = state_dict["BestValLoss"]
        best_model_state = state_dict["BestModel"]
        best_clsf_state  = state_dict["BestClsf"]
        optimizer.load_state_dict(state_dict["Optimizer"])
        print(f'Checkpoint loaded: best_vl_loss = {best_vl_loss:.4f}')
        if args.run_mode == 'unsupervised':
            print('Now continue the unsupervised training from the ckpt.')
        elif args.run_mode == 'finetune':
            print('Now continue the finetuning from the ckpt.')
        elif args.run_mode == 'test':
            print('Now do inferring using the ckpt.')
        print('-'*50)
    else:
        print('Not load checkpoints, begin finetuning from pretrained weights.')
        best_model_state = deepcopy(model.state_dict())
        best_clsf_state  = deepcopy(clsf .state_dict())

    if args.run_mode == 'test':
        ts_x_list, ts_y_list = get_data_dict[args.dataset](args, step='test')

        ts_logs, ts_loss = evaluate_epoch(args, ts_x_list, ts_y_list, model, clsf, loss_func, step='test')
        exit(0)

    # Settings to save ckpt
    if args.save_ckpt_path is not None:
        args.save_ckpt_path = f'{args.save_ckpt_path}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/{args.run_mode}_ckpt/'
        # create if not exist
        if not os.path.exists(args.save_ckpt_path):
            os.makedirs(args.save_ckpt_path)
        elif args.run_mode == 'finetune' and args.from_pretrained:  # not to continue finetuning, but begin fintuning
            shutil.rmtree(args.save_ckpt_path)
            os.mkdir(args.save_ckpt_path)
            print(f'To begin the finetuning, have deleted the existing save_ckpt_path in {args.save_ckpt_path}')

    print('-' * 50)

    print(f"Running at most {args.epoch_num} epochs")
    start_epoch = len(main_logs["epoch"]) if not args.from_pretrained else 0   # continue when restart
    last_tr_loss = 0
    wait_epoch = 0

    start_time = time.time()
    for epoch in range(start_epoch, args.epoch_num):
        print('-' * 50)
        print(f"Epoch {epoch}")
        # cpu_stats()

        tr_logs, tr_loss = train_epoch(args, tr_x_list, tr_y_list, model, clsf, loss_func, optimizer, scheduler,)
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
                f"{args.save_ckpt_path}/0.pt",
                # f"{args.save_ckpt_path}/{epoch}.pt",
            )
            save_logs(main_logs, f"{args.save_ckpt_path}/logs.json")
            print('Checkpoint and main logs saved.')

        if wait_epoch >= args.patience:
            break

    print('-' * 10, 'Training finished', '-' * 10)


