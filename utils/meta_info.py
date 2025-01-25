import torch
from torch import nn

from data_process.clinical_group_gene import clinical_group_data_gene
from data_process.defult_group_gene import default_group_data_gene
from data_process.get_data import default_get_data, clinical_get_data
from datasets.Brant1_dataset import Brant1_Dataset
from datasets.Brant2_dataset import Brant2_Dataset
from datasets.TFC_dataset import TFC_Dataset
from datasets.default_dataset import DefaultDataset
from model.BIOT.BIOT import BIOT_Trainer, BIOT
from model.BrainBERT.BrainBERT import BrainBERT_Trainer, BrainBERT
from model.Brant1.Brant1 import Brant1, Brant1_Trainer
from model.Brant2.Brant2 import Brant2_Trainer, Brant2
from model.GPT4TS.GPT4TS import GPT4TS_Trainer, GPT4TS
from model.LaBraM.LaBraM import LaBraM, LaBraM_Trainer
from model.SimMTM.SimMTM import SimMTM_Trainer, SimMTM
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
    'SeizureA': default_group_data_gene,
    'SeizureC': default_group_data_gene,
    'SeizureB': default_group_data_gene,
    'RepOD': default_group_data_gene,
    'UCSD_ON': default_group_data_gene,
    'UCSD_OFF': default_group_data_gene,
    'HUSM': default_group_data_gene,
    'ISRUC': default_group_data_gene,
}

get_data_dict = {
    'MAYO': default_get_data,
    'FNUSA': default_get_data,
    'CHBMIT': default_get_data,
    'Siena': default_get_data,
    'Clinical' : clinical_get_data,
    'SleepEDFx': default_get_data,
    'SeizureA': default_get_data,
    'SeizureC': default_get_data,
    'SeizureB': default_get_data,
    'RepOD': default_get_data,
    'UCSD_ON': default_get_data,
    'UCSD_OFF': default_get_data,
    'HUSM': default_get_data,
    'ISRUC': default_get_data,
}

metrics_dict = {
    'MAYO': BinaryClassMetrics,
    'FNUSA': BinaryClassMetrics,
    'CHBMIT': BinaryClassMetrics,
    'Siena': BinaryClassMetrics,
    'Clinical': BinaryClassMetrics,
    'SleepEDFx': MultiClassMetrics,
    'SeizureA': BinaryClassMetrics,
    'SeizureC': BinaryClassMetrics,
    'SeizureB': BinaryClassMetrics,
    'RepOD': BinaryClassMetrics,
    'UCSD_ON': BinaryClassMetrics,
    'UCSD_OFF': BinaryClassMetrics,
    'HUSM': BinaryClassMetrics,
    'ISRUC': MultiClassMetrics,
}

### To add a new method, please update the following dicts

dataset_class_dict = {
    'TFC': TFC_Dataset,
    'SimMTM': DefaultDataset,
    'BrainBERT': DefaultDataset,
    'GPT4TS': DefaultDataset,
    'Brant1': Brant1_Dataset,
    'Brant2': Brant2_Dataset,
    'LaBraM': DefaultDataset,
    'BIOT': DefaultDataset,
}

trainer_dict = {
    'TFC': TFC_Trainer,
    'SimMTM': SimMTM_Trainer,
    'BrainBERT': BrainBERT_Trainer,
    'GPT4TS': GPT4TS_Trainer,
    'Brant1': Brant1_Trainer,
    'Brant2': Brant2_Trainer,
    'LaBraM': LaBraM_Trainer,
    'BIOT': BIOT_Trainer,
}

model_dict = {
    'TFC': TFC,
    'SimMTM': SimMTM,
    'BrainBERT': BrainBERT,
    'GPT4TS': GPT4TS,
    'Brant1': Brant1,
    'Brant2': Brant2,
    'LaBraM': LaBraM,
    'BIOT': BIOT,
}


