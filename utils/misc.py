import random
import torch
import numpy as np
from copy import deepcopy
import sys
import time
import os
import psutil
import json


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def update_logs(logs, epo_loss, metrics=None):
    if metrics == None:     # unsupervised
        logs[f"Loss"] = epo_loss
    else:                   # finetune
        logs[f"Loss"] = epo_loss
        logs[f"Acc"] = metrics.acc * 100.0
        logs[f"Prec"] = metrics.prec * 100.0
        logs[f"Rec"] = metrics.rec * 100.0
        logs[f"F2"] = metrics.f_doub * 100.0
        logs[f"AUC"] = metrics.auc * 100.0

    return logs


def show_logs(prefix, logs, time_info):

    out = prefix + '  '
    for key in logs:
        out += f"{key}:{logs[key]:8.4f}  "
    out += time_info
    print(out)


def update_main_logs(logs, tr_logs, vl_logs, epoch):
    for key, value in dict(tr_logs, **vl_logs).items():
        if key not in logs:
            logs[key] = []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logs[key].append(value)
    logs["epoch"].append(epoch)
    return logs


# def batch_logs_update(task_type, logs, last_logs, loss, metrics,
#                       batch_id=None, log_interv=None, start_time=None, ):
#     if f"{task_type}_Loss" not in logs:
#         if loss.ndim == 0:
#             logs[f"{task_type}_Loss"] = np.zeros(1)
#             logs[f"{task_type}_Acc "] = np.zeros(1)
#             logs[f"{task_type}_Prec"] = np.zeros(1)
#             logs[f"{task_type}_Rec "] = np.zeros(1)
#             logs[f"{task_type}_F2  "] = np.zeros(1)
#             logs[f"{task_type}_AUC "] = np.zeros(1)
#         else:
#             # Multiple steps
#             logs[f"{task_type}_Loss"] = np.zeros(loss.size(0))
#             logs[f"{task_type}_Acc "] = np.zeros(metrics.acc.size(0))
#             logs[f"{task_type}_Prec"] = np.zeros(metrics.prec.size(0))
#             logs[f"{task_type}_Rec "] = np.zeros(metrics.rec.size(0))
#             logs[f"{task_type}_F2  "] = np.zeros(metrics.f_doub.size(0))
#             logs[f"{task_type}_AUC "] = np.zeros(metrics.auc.size(0))
#
#     logs[f"{task_type}_Loss"] += loss.detach().cpu().numpy()
#     logs[f"{task_type}_Acc "] += metrics.acc
#     logs[f"{task_type}_Prec"] += metrics.prec
#     logs[f"{task_type}_Rec "] += metrics.rec
#     logs[f"{task_type}_F2  "] += metrics.f_doub
#     logs[f"{task_type}_AUC "] += metrics.auc
#
#     if batch_id is not None:
#         if (batch_id + 1) % log_interv == 0:
#             elapsed = time.perf_counter() - start_time
#             loc_logs = update_logs(logs, log_interv, last_logs)
#             last_logs = deepcopy(logs)
#
#             show_logs(f"Batch 0~{batch_id + 1} used {elapsed:.1f}s. {1000.0 * elapsed / log_interv:.1f}ms per batch.", loc_logs)
#         return last_logs
#     return logs


def save_logs(data, path_logs):
    with open(path_logs, 'w') as file:
        json.dump(data, file, indent=2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def load_checkpoint(load_path):
    # Load specific checkpoint
    if not os.path.isdir(load_path):
        if load_path.split('.')[-1] == 'pt':
            checkpoint_file = load_path
            load_path = os.path.split(load_path)[0]
        else:
            print("Invalid checkpoints path at " + load_path)
            return None
    # If no specific checkpoint is assigned, load the newest checkpoint
    else:
        checkpoints = [x for x in os.listdir(load_path)
                       if os.path.splitext(x)[1] == '.pt'
                       and os.path.splitext(x)[0][-1].isdigit()]
        if len(checkpoints) == 0:
            print("No checkpoints found at " + load_path)
            return None
        checkpoints.sort(key=lambda x: int(os.path.splitext(x)[0][-1]))
        checkpoint_file = os.path.join(load_path, checkpoints[-1])

    with open(os.path.join(load_path, 'logs.json'), 'rb') as file:
        logs = json.load(file)

    return os.path.abspath(checkpoint_file), logs


def save_checkpoint(model_state, optimizer_state, best_model_state, best_val_loss, path_checkpoint):
    state_dict = {"Model": model_state,
                  "Optimizer": optimizer_state,
                  "BestModel": best_model_state,
                  "BestValLoss": best_val_loss}

    torch.save(state_dict, path_checkpoint)
