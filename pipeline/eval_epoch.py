import time

import numpy as np
import torch
from tqdm import tqdm

from data_process.data_info import data_info_dict
from utils.meta_info import dataset_class_dict, metrics_dict
from utils.misc import update_logs, show_logs, make_dir_if_not_exist
from utils.config import config

def evaluate_epoch(args, x_list, y_list, model, clsf, loss_func, step):
    assert step == 'valid' or step == 'test'

    model.eval()
    clsf.eval()
    device = next(model.parameters()).device

    start_time = time.perf_counter()
    epo_logs = {}
    batch_cnt = 0
    epo_loss = 0

    if args.run_mode == 'finetune' or args.run_mode == 'test':
        epo_y = torch.tensor([], dtype=torch.long)
        epo_pred = torch.tensor([], dtype=torch.long)
        epo_logit = torch.tensor([], dtype=torch.float32)

    file_num = len(x_list)
    for file_idx in range(file_num):
        x = x_list[file_idx]
        y = y_list[file_idx]

        valid_dataset = dataset_class_dict[args.model](args, x, y)
        valid_loader = valid_dataset.get_data_loader(args.batch_size, shuffle=True, num_workers=0)

        if args.run_mode == 'finetune' or args.run_mode == 'test':
            file_y = torch.tensor([], dtype=torch.long)
            file_pred = torch.tensor([], dtype=torch.long)
            file_logit = torch.tensor([], dtype=torch.float32)

        for batch_id, data_packet in enumerate(tqdm(valid_loader, disable=args.tqdm_dis, desc=f'file{file_idx}/{file_num}')):
            # x: (bsz, ch_num, seq_len, patch_len)
            # y: (bsz, )
            data_packet = [d.to(device) for d in data_packet]

            with torch.no_grad():
                if args.run_mode == 'unsupervised':
                    batch_loss = model.forward_propagate(args, data_packet,
                                                         model, clsf, loss_func)
                    epo_loss += batch_loss.detach().cpu().numpy()

                elif args.run_mode == 'finetune':
                    batch_loss, logit, y = model.forward_propagate(args, data_packet,
                                                                   model, clsf, loss_func)
                    epo_loss += batch_loss.detach().cpu().numpy()

                elif args.run_mode == 'test':
                    logit, y = model.forward_propagate(args, data_packet,
                                                       model, clsf)

                if args.run_mode == 'finetune' or args.run_mode == 'test':
                    pred = torch.argmax(logit, dim=-1)
                    file_y     = torch.cat([file_y,     y.cpu()], dim=0)
                    file_pred  = torch.cat([file_pred,  pred.detach().cpu()], dim=0)
                    file_logit = torch.cat([file_logit, logit.detach().cpu()], dim=0)

                    epo_y     = torch.cat([epo_y,     file_y.cpu()], dim=0)
                    epo_pred  = torch.cat([epo_pred,  file_pred.detach().cpu()], dim=0)
                    epo_logit = torch.cat([epo_logit, file_logit.detach().cpu()], dim=0)

            batch_cnt += 1

        if step == 'test' and data_info_dict[args.dataset]['label_level'] == 'channel_level':
            save_logit_path = f'{config["test_log_path"]}/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/'
            make_dir_if_not_exist(save_logit_path)
            # logit: (seq_num*ch_num, fake_ch_num=1, 2)
            # y: (seq_num*ch_num)
            file_logit_save = np.reshape(np.array(file_logit), (-1, valid_dataset.ch_num, 2))
            file_y_save     = np.reshape(np.array(file_y),     (-1, valid_dataset.ch_num, ))
            np.save(f'{save_logit_path}/file{file_idx}_logit.npy', file_logit_save)
            np.save(f'{save_logit_path}/file{file_idx}_y.npy',     file_y_save)

        valid_dataset.reload_pool.close()

    if args.run_mode == 'finetune' or args.run_mode == 'test':
        metrics = metrics_dict[args.dataset](args, epo_pred.cpu(), epo_logit.cpu(), epo_y.cpu())
    else:
        metrics = None

    epo_loss /= batch_cnt
    epo_logs = update_logs(args, epo_logs, epo_loss, metrics)

    if step == 'valid':
        show_logs('[Valid]', epo_logs, f"{(time.perf_counter()-start_time):.1f}s")
        if args.run_mode == 'finetune':
            print(metrics.conf_matrix)
    elif step == 'test':
        show_logs('[Test]', epo_logs, None)
        print(metrics.conf_matrix)

    return epo_logs, epo_loss
