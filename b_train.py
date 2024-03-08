import random

import torch
import numpy as np
import time
import os
import sys
import argparse
import json
import shutil
from copy import deepcopy

from data_process.get_dataset import get_dataset

from model.ch_aggr_clsf import ChannelAggrClsf
from utils.meta_info import dataset_dict, model_dict, forward_dict, set_model_config_dict, loss_func_dict, metrics_dict
from utils.misc import set_seed, load_checkpoint, cpu_stats, update_logs, show_logs, \
    save_logs, save_checkpoint, update_main_logs


def train_epoch(args, dataloader, model, clsf, loss_func, optimizer, ):
    model.train()
    clsf.train()
    device = next(model.parameters()).device

    start_time = time.perf_counter()
    logs, last_logs = {}, None
    batch_cnt = 0
    epo_loss = 0

    if args.train_mode == 'finetune':
        epo_y     = torch.tensor([], dtype=torch.long)
        epo_pred  = torch.tensor([], dtype=torch.long)
        epo_logit = torch.tensor([], dtype=torch.float32)

    for batch_id, data_packet in enumerate(dataloader):
        # x: (bsz, ch_num, seq_len, patch_len)
        # y: (bsz, )
        # x = x.to(device)
        # y = y.to(device)
        data_packet = [d.to(device) for d in data_packet]

        ret = forward_dict[args.model](args, data_packet, model, clsf, loss_func)

        if args.train_mode == 'unsupervised':
            batch_loss = ret
        elif args.train_mode == 'finetune':
            batch_loss, logit, y = ret

            pred = torch.argmax(logit, dim=-1)
            epo_y = torch.cat([epo_y, y.cpu()])
            epo_pred = torch.cat([epo_pred, pred.detach().cpu()], dim=0)
            epo_logit = torch.cat([epo_logit, logit.detach().cpu()], dim=0)
        else:
            raise NotImplementedError(f'Undefined training mode {args.train_mode}')

        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_cnt += 1
        epo_loss += batch_loss.detach().cpu().numpy()

    optimizer.step()
    optimizer.zero_grad()

    if args.train_mode == 'finetune':
        metrics = metrics_dict[args.dataset](epo_pred.cpu(), epo_logit.cpu(), epo_y.cpu())
    else:
        metrics = None

    logs = update_logs(logs, epo_loss, metrics)
    show_logs('[Train]', logs, f"{(time.perf_counter()-start_time):.1f}s")

    return logs, epo_loss



def valid_epoch(args, dataloader, model, clsf, loss_func, ):
    model.eval()
    clsf.eval()
    device = next(model.parameters()).device

    start_time = time.perf_counter()
    logs, last_logs = {}, None
    batch_cnt = 0
    epo_loss = 0

    if args.train_mode == 'finetune':
        epo_y = torch.tensor([], dtype=torch.long)
        epo_pred = torch.tensor([], dtype=torch.long)
        epo_logit = torch.tensor([], dtype=torch.float32)

    for batch_id, data_packet in enumerate(dataloader):
        # x: (bsz, ch_num, seq_len, patch_len)
        # y: (bsz, )
        # x = x.to(device)
        # y = y.to(device)
        data_packet = [d.to(device) for d in data_packet]

        with torch.no_grad():
            ret = forward_dict[args.model](args, data_packet, model, clsf, loss_func)

            if args.train_mode == 'unsupervised':
                batch_loss = ret
            elif args.train_mode == 'finetune':
                batch_loss, logit, y = ret

                pred = torch.argmax(logit, dim=-1)
                epo_y = torch.cat([epo_y, y.cpu()])
                epo_pred = torch.cat([epo_pred, pred.detach().cpu()], dim=0)
                epo_logit = torch.cat([epo_logit, logit.detach().cpu()], dim=0)
            else:
                raise NotImplementedError(f'Undefined training mode {args.train_mode}')

        batch_cnt += 1
        epo_loss += batch_loss.detach().cpu().numpy()

    if args.train_mode == 'finetune':
        metrics = metrics_dict[args.dataset](epo_pred.cpu(), epo_logit.cpu(), epo_y.cpu())
    else:
        metrics = None

    logs = update_logs(logs, epo_loss, metrics)
    show_logs('[Valid]', logs, f"{(time.perf_counter()-start_time):.1f}s")

    return logs, epo_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BrainBenchmark')

    group_train = parser.add_argument_group('Train')
    group_train.add_argument('--exp_id', type=str, default='-1',
                             help='The experimental id.')
    group_train.add_argument('--gpu_id', type=int, default=2,
                             help='The gpu id.')
    group_train.add_argument('--train_mode', type=str, default='finetune',   # unsupervised, finetune
                             help='To perform self-/unsupervised training, or finetuning.')
    group_train.add_argument('--batch_size', type=int, default=128,
                             help='Number of batches.')
    group_train.add_argument('--save_step', type=int, default=5,
                             help='The step number to save checkpoint.')
    group_train.add_argument('--epoch_num', type=int, default=100,
                             help='Epoch number for total iterations.')
    group_train.add_argument('--loss_change', type=float, default=1e-3,
                             help='The convergence tolerance value to stop training.')
    group_train.add_argument('--early_stop', action='store_false',
                             help='Whether to use early stopping technique during training.')
    group_train.add_argument('--patience', type=int, default=10,
                             help='The waiting epoch number for early stopping.')
    group_train.add_argument('--load_ckpt_path', type=str, default=None,    # None '/data/brainnet/benchmark/ckpt/'
                             help='The path to load checkpoint (.pt file or the upper path).')
    group_train.add_argument('--load_best', action='store_false',
                             help='Whether to load the best state in the checkpoints.')
    group_train.add_argument('--save_ckpt_path', type=str, default='/data/brainnet/benchmark/ckpt/',
                             help='The path to save checkpoint')
    group_train.add_argument('--lr', type=float, default=3e-4,
                             help='The learning rate.')
    group_train.add_argument('--weight_decay', type=float, default=1e-4,
                             help='The weight decay.')
    group_train.add_argument('--gpu', action='store_false',
                             help='Whether to load the data and model to GPU.')

    group_data = parser.add_argument_group('Data')
    group_data.add_argument('--data_load_dir', type=str, default='/data/brainnet/benchmark/gene_data/',
                            help='The path to load the generated data.')
    group_data.add_argument('--sample_seq_num', type=float, default=2400,
                            help='The number of sequence sampled from each dataset.')
    group_data.add_argument('--seq_len', type=float, default=16,
                            help='The number of patches in a sequence.')
    group_data.add_argument('--patch_len', type=float, default=500,
                            help='The number of points in a patch.')
    group_data.add_argument('--dataset', type=str, default='MAYO',
                            help='The dataset to perform training.')

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

    args = set_model_config_dict[args.model](parser)
    args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
        args.dataset,
        args.sample_seq_num,
        args.seq_len,
        args.patch_len,
    )

    if args.random_seed is None:
        args.random_seed = random.randint(0, 2 ** 31)
    set_seed(args.random_seed)

    if args.gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    main_logs = {"epoch": []}

    train_dataset = get_dataset(args, step='train')
    valid_dataset = get_dataset(args, step='valid')

    train_loader = train_dataset.get_data_loader(args.batch_size, shuffle=True, num_workers=0)
    valid_loader = valid_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)

    args.n_class = train_dataset.n_class

    # host_name = os.getcwd().split('/')[2]
    # args.path_checkpoint = utils.paths[host_name]  # 根据用户设置path_checkpoint

    model = model_dict[args.model](args).to(device)
    clsf  = ChannelAggrClsf(args).to(device)

    loss_func = loss_func_dict[args.model]()

    optimizer = torch.optim.AdamW([
         {'params': list(model.parameters()), 'lr': args.lr},
         {'params': list(clsf .parameters()), 'lr': args.lr},
         ],
        betas=(0.9, 0.95), eps=1e-5,
    )

    print(f'CONFIG:\n{json.dumps(vars(args), indent=4, sort_keys=True)}')
    print('-' * 50)

    best_vl_loss = np.inf
    state_dict = None

    # Load ckpt
    if args.load_ckpt_path is not None:
        args.load_ckpt_path = f'{args.load_ckpt_path}/{args.model}_exp{args.exp_id}_{args.data_id}/'
        load_path, main_logs = load_checkpoint(args.load_ckpt_path)
        print('Load checkpoint:', load_path)

        state_dict = torch.load(load_path, 'cpu')
        if args.load_best:
            model.load_state_dict(state_dict["BestModel"], strict=False)
        else:
            model.load_state_dict(state_dict["Model"], strict=False)
        best_vl_loss = state_dict["BestValLoss"]
        best_model_state = state_dict["BestModel"]
        optimizer.load_state_dict(state_dict["Optimizer"])
        print(f'Checkpoint loaded: best_vl_loss = {best_vl_loss:.4f}')
    else:
        print('Not load checkpoint')
        best_model_state = deepcopy(model.state_dict())

    # Settings to save ckpt
    if args.save_ckpt_path is not None:
        args.save_ckpt_path = f'{args.save_ckpt_path}/{args.model}_exp{args.exp_id}_{args.data_id}/'
        # create if not exist, delete if exist
        if not os.path.exists(args.save_ckpt_path):
            os.makedirs(args.save_ckpt_path)
        # else:
        #     shutil.rmtree(args.save_ckpt_path)
        #     os.mkdir(args.save_ckpt_path)
        #     print(f'Have delete and re-create the save_ckpt_path: {args.save_ckpt_path}')

    print('-' * 50)

    print(f"Running at most {args.epoch_num} epochs")
    start_epoch = len(main_logs["epoch"])    # continue when restart
    last_loss = 0
    wait_epoch = 0

    start_time = time.time()
    for epoch in range(start_epoch, args.epoch_num):
        print('-' * 50)
        print(f"Epoch {epoch}")
        # cpu_stats()

        tr_logs, tr_loss = train_epoch(args, train_loader, model, clsf, loss_func, optimizer,)

        vl_logs, vl_loss = valid_epoch(args, valid_loader, model, clsf, loss_func,)

        # print(f'Ran {epoch - start_epoch + 1} epochs in {time.time() - start_time:.2f} seconds')

        # process training loss
        loss_change = np.fabs(tr_loss - last_loss)
        last_loss = tr_loss
        # process validation loss
        if vl_loss < best_vl_loss:
            best_vl_loss = deepcopy(vl_loss)
            best_model_state = deepcopy(model.state_dict())
            print('Best model state updated.')
            wait_epoch = 0
        elif args.early_stop:
            wait_epoch += 1

        # update main logs
        main_logs = update_main_logs(main_logs, tr_logs, vl_logs, epoch)

        # save checkpoint at every epoch
        if args.save_ckpt_path is not None and (
                epoch % args.save_step == 0 or epoch == args.epoch_num - 1 or
                loss_change <= args.loss_change or wait_epoch >= args.patience):
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()

            save_checkpoint(
                model_state,
                optimizer_state,
                best_model_state,
                best_vl_loss,
                f"{args.save_ckpt_path}/{args.train_mode}_{epoch}.pt",
            )
            save_logs(main_logs, f"{args.save_ckpt_path}/{args.train_mode}_logs.json")
            print('Checkpoint and main logs saved.')

        if loss_change <= args.loss_change or wait_epoch >= args.patience:
            break

    train_dataset.reload_pool.close()
    valid_dataset.reload_pool.close()

    print('-' * 10, 'Training finished', '-' * 10)


