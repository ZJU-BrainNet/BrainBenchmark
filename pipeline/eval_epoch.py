import time

import torch
from tqdm import tqdm

from utils.meta_info import dataset_class_dict, metrics_dict
from utils.misc import update_logs, show_logs


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
        valid_loader = valid_dataset.get_data_loader(args.batch_size, shuffle=False, num_workers=0)

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
                    epo_y = torch.cat([epo_y, y.cpu()])
                    epo_pred = torch.cat([epo_pred, pred.detach().cpu()], dim=0)
                    epo_logit = torch.cat([epo_logit, logit.detach().cpu()], dim=0)

            batch_cnt += 1

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

    return epo_logs, epo_loss
