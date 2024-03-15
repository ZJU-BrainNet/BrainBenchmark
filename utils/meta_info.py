import torch
from torch import nn

from data_process.clinical_group_gene import clinical_group_data_gene
from data_process.defult_group_gene import default_group_data_gene
from data_process.get_data import default_get_data, clinical_get_data
from datasets.TFC_dataset import TFC_Dataset
from datasets.default_dataset import DefaultDataset
# from model.BrainBERT.BrainBERT import set_BrainBERT_config, load_pretrained_weights_BrainBERT, BrainBERT, \
    # forward_BrainBERT, adapt_to_input_size_BrainBERT, clsf_loss_func_BrainBERT
# from model.TFC.TFC import forward_TFC, TFC, set_TFC_config
from model.BrainBERT.BrainBERT import BrainBERT_Trainer, BrainBERT
from model.TFC.TFC import TFC_Trainer, TFC
from utils.metrics import BinaryClassMetrics, MultiClassMetrics

### To add a new dataset, please update the following dicts

group_data_gene_dict = {
    'MAYO' : default_group_data_gene,
    'FNUSA' : default_group_data_gene,
    'CHBMIT' : default_group_data_gene,
    'Siena' : default_group_data_gene,
    'Clinical' : clinical_group_data_gene,
    'SleepEDFx' : default_group_data_gene,
}

get_data_dict = {
    'MAYO': default_get_data,
    'FNUSA': default_get_data,
    'CHBMIT': default_get_data,
    'Siena': default_get_data,
    'Clinical' : clinical_get_data,
    'SleepEDFx': default_get_data,
}

metrics_dict = {
    'MAYO': BinaryClassMetrics,
    'FNUSA': BinaryClassMetrics,
    'CHBMIT': BinaryClassMetrics,
    'Siena': BinaryClassMetrics,
    'Clinical': BinaryClassMetrics,
    'SleepEDFx': MultiClassMetrics,
}

### To add a new method, please update the following dicts

# set_model_config_dict = {
#     'TFC': set_TFC_config,
#     'BrainBERT': set_BrainBERT_config,
# }

dataset_class_dict = {
    'TFC': TFC_Dataset,
    'BrainBERT': DefaultDataset,
}

trainer_dict = {
    'TFC': TFC_Trainer,
    'BrainBERT': BrainBERT_Trainer,
}

model_dict = {
    'TFC': TFC,
    'BrainBERT': BrainBERT,
}

# loss_func_dict = {
#     'TFC': nn.CrossEntropyLoss,
#     'BrainBERT': clsf_loss_func_BrainBERT,
#
#
# }

# forward_dict = {
#     'TFC': forward_TFC,
#     'BrainBERT': forward_BrainBERT,
# }

### For the pretrained methods, please update the following dicts

# pretrained_model_info_dict = {
#     'BrainBERT': {
#         'load_weights_func': load_pretrained_weights_BrainBERT,
#         'adapt_input':adapt_to_input_size_BrainBERT,
#     }
# }


