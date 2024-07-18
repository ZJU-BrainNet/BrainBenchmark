import sys

import argparse
import numpy as np

from utils.metrics import BinaryClassMetrics

parser = argparse.ArgumentParser(description='')
group = parser.add_argument_group('Train')

group.add_argument('--exp_id', type=str, default='-1',
                         help='The experimental id.')
group.add_argument('--cv_id', type=int, default=3,
                         help='The cross validation id.')
group.add_argument('--run_mode', type=str, default='test',  # finetune, test
                         help='To perform finetuning, or testing.')

group.add_argument('--model', type=str, default='Brant1',  # BrainBERT GPT4TS Brant1
                        help='The model to run.')

group.add_argument('--dataset', type=str, default='Clinical',
                        help='The dataset to perform training.')
group.add_argument('--sample_seq_num', type=int, default=None,
                        help='The number of sequence sampled from each dataset.')
group.add_argument('--seq_len', type=int, default=15,
                        help='The number of patches in a sequence.')
group.add_argument('--patch_len', type=int, default=1500,
                        help='The number of points in a patch.')

argv = sys.argv[1:]
args = parser.parse_args(argv)


args.data_id = '{}_ssn{}_sl{}_pl{}'.format(
    args.dataset,
    args.sample_seq_num,
    args.seq_len,
    args.patch_len,
)

save_logit_path = f'/data/brainnet/benchmark/test_logits/{args.model}_exp{args.exp_id}_cv{args.cv_id}_{args.data_id}/'



subj_y, subj_pred, subj_logit = [], [], []
chan_y, chan_pred, chan_logit = [], [], []

for file_idx in range(2):
    file_logit = np.load(f'{save_logit_path}/file{file_idx}_logit.npy')  # (seq_num, ch_num, 2)
    file_y     = np.load(f'{save_logit_path}/file{file_idx}_y.npy')      # (seq_num, ch_num)
    file_pred  = np.argmax(file_logit, axis=-1)

    ### channel-level
    chan_y.append(file_y)
    chan_pred.append(file_pred)
    chan_logit.append(file_logit)

    ### subject-level
    threshold = 1

    # aggregate to subject-level
    file_pred = np.sum(file_pred, axis=-1)  # (seq_num,)
    file_pred[file_pred <  threshold] = 0
    file_pred[file_pred >= threshold] = 1

    file_logit = np.mean(file_logit, axis=-2)  # (seq_num, 2)
    file_y = np.max(file_y, axis=-1)

    subj_y.append(file_y)
    subj_pred.append(file_pred)
    subj_logit.append(file_logit)


chan_y     = np.concatenate(chan_y, axis=0)
chan_pred  = np.concatenate(chan_pred, axis=0)
chan_logit = np.concatenate(chan_logit, axis=0)

subj_y     = np.concatenate(subj_y, axis=0)
subj_pred  = np.concatenate(subj_pred, axis=0)
subj_logit = np.concatenate(subj_logit, axis=0)


# channel-level
metrics = BinaryClassMetrics(args, chan_pred, chan_logit, chan_y)
print('channel level')
print(metrics.get_metrics())
print(metrics.get_confusion())

# subject-level
metrics = BinaryClassMetrics(args, subj_pred, subj_logit, subj_y)
print('\nsubject level')
print(metrics.get_metrics())
print(metrics.get_confusion())


# inverse the prediction
_1_pos = np.where(subj_pred==1)[0]
_0_pos = np.where(subj_pred==0)[0]
subj_pred[_1_pos] = 0
subj_pred[_0_pos] = 1

subj_logit = subj_logit[:, ::-1]    # 交换两个预测结果

# subject-level
metrics = BinaryClassMetrics(args, subj_pred, subj_logit, subj_y)
print('\ninverse the prediction: subject level')
print(metrics.get_metrics())
print(metrics.get_confusion())