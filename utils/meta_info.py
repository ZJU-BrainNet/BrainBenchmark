from torch import nn

from data_process.data_gene import DataGene_MAYO, DataGene_FNUSA
from datasets.TFC_dataset import TFC_Dataset
from datasets.default_dataset import DefaultDataset
from model.TFC.TFC import forward_TFC, TFC, set_TFC_config
from utils.metrics import BinaryClassMetrics

dataset_dict = {
    'MAYO' : DataGene_MAYO,
    'FNUSA' : DataGene_FNUSA,
    # 'CHBMIT' : DataGene_CHBMIT,
    # 'Siena' : DataGene_Siena,
    # 'Clinical' : DataGene_Clinical,
    # 'SleepEDFx' : DataGene_SleepEDFx,
    # 'HMC' : DataGene_HMC,
    # 'MotorImg' : DataGene_MotorImg,
}

set_model_config_dict = {
    'TFC': set_TFC_config,
    # 'PatchTST': set_PatchTST_config,
}

dataset_class_dict = {
    'xxx': DefaultDataset,
    'TFC': TFC_Dataset,
}

model_dict = {
    'TFC': TFC,
    # 'MBrain' : MBrain,
    # 'BrainBERT' : BrainBERT,
}

loss_func_dict = {
    'TFC': nn.CrossEntropyLoss,

}

forward_dict = {
    'TFC': forward_TFC,
    # 'MBrain': forward_MBrain,
    # 'BrainBERT': forward_BrainBERT,
}

metrics_dict = {
    'MAYO': BinaryClassMetrics,
}
