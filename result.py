import os
import json
import pandas as pd
import re


def read_one_file(file_path, ):
    file_name = os.path.join(file_path, 'finetune_ckpt', 'logs.json')
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'r') as f:
        data_dict = json.load(f)
    dict_len = len(data_dict['epoch'])
    for key in data_dict:
        if len(data_dict[key]) < dict_len:
            dict_len = len(data_dict[key])
    # "epoch", "Loss", "Acc", "Prec", "Rec", "F2", "AUPRC", "AUROC", 
    max_dict = {'Loss': 1000.0}
    for i, loss in enumerate(data_dict['Loss']):
        if i >= dict_len:
            break
        if loss < max_dict['Loss']:
            for key in data_dict:
                max_dict[key] = data_dict[key][i]
    return max_dict


if __name__ == "__main__":
    root_path = '/data/share/benchmark/ckpt'
    files = os.listdir(root_path)
    result_dict = {'model_name': [], 'exp_num': [], 'cross_validation_num': [], 'dataset': [],
                   'sample_seq_num': [], 'seq_len': [], 'patch_len': [], 'Acc': [], 'Prec': [],
                   'Rec': [], 'F2': [], 'AUPRC': [], 'AUROC': [], 'TopKAcc': [], 'Sens': [],
                   'Spec': [], 'MF1': [], 'Kappa': []}
    for num, file in enumerate(files):
        one_dict = read_one_file(os.path.join(root_path, file))
        if one_dict == None:
            continue
        information = file.split('_')
        # BIOT_exp-1_cv0_CHBMIT_ssnNone_sl8_pl128
        result_dict['model_name'].append(information[0])
        result_dict['exp_num'].append(information[1][-1])
        result_dict['cross_validation_num'].append(information[2][-1])
        result_dict['dataset'].append(information[3])
        result_dict['sample_seq_num'].append("".join(re.findall(r'\d+', information[4])))
        result_dict['seq_len'].append("".join(re.findall(r'\d+', information[5])))
        result_dict['patch_len'].append("".join(re.findall(r'\d+', information[6])))
        for key in list(result_dict.keys())[7:]:
            if key in one_dict:
                result_dict[key].append(one_dict[key])
            else:
                result_dict[key].append('')
                
    df = pd.DataFrame(result_dict)
    df.to_excel('result.xlsx', index=False)