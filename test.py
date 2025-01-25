import os
import re
import pandas as pd
import subprocess
import torch

num_threads = '32'
torch.set_num_threads(int(num_threads))
os.environ['OMP_NUM_THREADS'] = num_threads
os.environ['OPENBLAS_NUM_THREADS'] = num_threads
os.environ['MKL_NUM_THREADS'] = num_threads
os.environ['VECLIB_MAXIMUM_THREADS'] = num_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_threads

fold_lst = os.listdir('/data/share/benchmark/ckpt')
df = pd.read_excel('ch_num.xlsx')
for folder in fold_lst:
    if 'Seizure' in folder or 'Clinic' in folder:
        continue
    items = folder.split('_')
    model_name = items[0]
    dataset = items[3]
    cv = int("".join(re.findall(r'\d+', items[2])))
    if 'ON' in folder or 'OFF' in folder:
        dataset += '_' + items[4]
        sl = int("".join(re.findall(r'\d+', items[6])))
    else:
        sl = int("".join(re.findall(r'\d+', items[5])))
    pl = int("".join(re.findall(r'\d+', items[6])))
    ch_num = df.loc[df['dataset'] == dataset, 'ch_num'].iloc[0]
    cmd3 = f'CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 python pretrained_run.py --run_mode test --dataset {dataset} --model {model_name} --seq_len {sl} --patch_len {pl} --cnn_in_channels {ch_num} --cv_id {cv}'
    result = subprocess.run(cmd3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)