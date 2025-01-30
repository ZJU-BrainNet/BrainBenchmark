# BrainBenchmark
The benchmark of self-supervised/unsupervised and pretrained models on brain signals. 



# How to start

Please install the following requirements:
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.2.2
scipy==1.10.1
torch==2.0.1
tqdm==4.64.1
```

## Data Generating 

For each dataset you want to run experiments on, the first thing to do is generating a specific set of data on your device. Just run the `data_gene.py`, passing in the specific argument value: `dataset`, `sample_seq_num`, `seq_len`, and  `patch_len`.

These four arguments can determine one specific set of data. 

## Training and Evaluation

This benchmark support the training and evaluating of two kinds of methods: 1) The models with a pretrained checkpoint; 2) An self-supervised or unsupervised without checkpoints. 

### Pretrained Models

To load a checkpoint and train or evaluate from the checkpoint, please run the `pretrained_run.py`. For the `--run_mode` parameter, you can choose one from these strings:

- `finetune`: load the checkpoint, and begin finetune from the checkpoint.
- `test`: evaluate the model with this checkpoint.

### Self-/unsupervised Methods

To perform the self-supervised or unsupervised training, please run the `unsupervised_run.py`. For the `run_mode` argument, you can choose one from these strings:

- `unsupervised`: begin the self-supervised or unsupervised training process and save some checkpoints. Then, your checkpoints may be loaded using the `finetune` as the value of `run_mode` argument. 
- `finetune`: load the checkpoint, and begin finetune from the checkpoint.
- `test`: evaluate the model with this checkpoint.

### Note

**Fair comparison.** If you want to evaluate a result and make a direct performance comparison with other models on the same dataset, the following arguments about input data must be set according to a unified setting. These arguments includes `sample_seq_num`, `seq_len`, `patch_len`. 

**Channel fusion.** Since many methods can only handle single-channel data, an *inter-channel CNN* is used to aggregate the representations across all channels to obtain a "subject-level" representation. The model architecture of this CNN can be set with the following arguments: `cnn_in_channels`, `cnn_kernel_size`. 

**Loading checkpoints.** If you need to load ckpt (continue training from the last breakpoint), please add the `load_ckpt_path` argument (`None` if train from scratch). The path to save model checkpoints can also be set with the `save_ckpt_path` argument. 

```
--load_ckpt_path f'{config["ckpt_path"]}/' 
--save_ckpt_path f'{config["ckpt_path"]}/' 
```



# How to extend

## To add a new dataset

1. Split all the subjects in the new dataset into several groups (4-6 groups are recommended). Each group of data should be generated as a signle file named like `group_0_data.npy`, ..., `group_5_data.npy`. In each file, the shape of the numpy array is: `(seq_num, ch_num, seq_len, patch_len)` 

   The cooresponding label files should be named in similar format: `group_0_label.npy`, ..., `group_5_label.npy`. 

2. Then, add a new element in the `data_info_dict` from  `BrainBenchmark/data_process/data_info.py`. Taking MAYO as an example: 

   ```python
   'MAYO': {'data_path': '/data/brainnet/public_dataset/MAYO/group_data_15000_2to0/',
       'group_num': 6,
       'split': [3, 1, 2],
       'various_ch_num': False,
       'n_class': 2,
       'label_level': 'subject_level',
       },
   ```

   - `split`: how to split the 6 groups, as training/validation/testing respectively.
   - `various_ch_num`: whether or not the channel number may varies between different data files in this dataset.
   - `n_class`: the task performed on this dataset is a n-class classification task.
   - `label_level`: the labels are subject-level or channel-level. 

3. For most of datasets, the label are subject-level without any variations in channel number. In this case, you can usually use the `default_group_data_gene()` function in the `BrainBenchmark/data_process/default_group_gene.py`. For other special cases like our clinical datasets, please refer to `clinical_group_gene.py` for reference. 

4. In the `BrainBenchmark/utils/meta_info.py`, assume the new dataset name is `NAME`, 

   - Add a line `'NAME': <group_data_gene>,` to the dictionary `group_data_gene_dict`, and import the function `<group_data_gene>` here. 
   - Add a line `'NAME': default_get_data,` to the dictionary `get_data_dict`. 
   - Add a line `'NAME': BinaryClassMetrics, ` (if `n_class==2`) or `'NAME': MultiClassMetrics, ` (if `n_class>=3`)  to the dictionary `metrics_dict`. 

## To add a new method

Assume that the method name is `NAME`, 

1. Make a new directory `BrainBenchmark/model/NAME/`. 

2. Make a new file named `NAME.py` here, and write two classes in this file: `NAME_Trainer` and `NAME`. 

   The class `NAME_Trainer` must includes these functions as members:

   - `def set_config(args: Namespace)` : A static method that sets all of the method's unique parameters as input arguments, such that any user can set these arguments. Taking TF-C model as an example:  

   ```python
       @staticmethod
       def set_config(args: Namespace):
         args.Context_Cont_temperature = 0.2
         args.Context_Cont_use_cosine_similarity = True
         args.augmentation_max_seg = 12
         args.augmentation_jitter_ratio = 2
         args.augmentation_jitter_scale_ratio = 1.5
         return args
   ```

   - `def clsf_loss_func(args)` : A static method that returns the loss function used by this method. Taking TF-C model as an example:  

   ```python
       @staticmethod
       def clsf_loss_func(args):
           return nn.CrossEntropyLoss()
   ```

   - `def optimizer(args, model, clsf) ` : A static method that returns the optimizer used by this method. Taking TF-C model as an example:

   ```python
       @staticmethod
       def optimizer(args, model, clsf):
           return torch.optim.AdamW([
               {'params': list(model.parameters()), 'lr': args.lr},
               {'params': list(clsf.parameters()), 'lr': args.lr},
           ],
               betas=(0.9, 0.95), eps=1e-5,
           )
   ```

   The class `NAME` must includes these functions as members:

   - `def forward_propagate(args, data_packet, model, clsf, loss_func=None)` : based on the data batch `data_packet` (this is determined by the `NAME_dataset` you write later), write the code for model forward propagation and loss calculation. If the code is different between the self-/unsupervision phase and fine-tuning phase, you can use the argument `args.run_mode`  to branch. Taking TF-C model as an example:

   ```python
       @staticmethod
       def forward_propagate(args, data_packet, model, clsf, loss_func=None):
           # x: (bsz, ch_num, seq_len, patch_len)
           # y: (bsz, )
           device = next(model.parameters()).device
   
           if args.train_mode == "unsupervised":
               x, aug1_x, f, aug1_f = data_packet
               # code to perform self-supervised training
               return loss
   
           elif args.train_mode == "finetune":
               x, y, aug1_x, f, aug1_f = data_packet
               # code to perform fine-tuning
               return loss, logit, y
   
           else:
               raise NotImplementedError(f'Undefined training mode {args.train_mode}')
   ```

3. Then add any other files about your model in the directory `BrainBenchmark/model/NAME/` to implement the method. 

4. For some methods, they require unique data process (like calculating the spectral density and so on), therefore this benchmark supports to add any new Dataset class for a new method. 

   Make a new file `BrainBenchmark/datasets/NAME_dataset.py`, and write your dataset class `NAME_Dataset` here. Please make sure that the data tuple returned in the `__getitem__` function matches what you receive in the `forward_propagate` function. Make sure that your class contains the following basic member functions: `__len__`, `get_data_loader`. Taking `TF-C_Dataset` as an example: 

   ```python
   class TF-C_Dataset(Dataset):
       def __init__(self, args, x, y):
           # x: (seq_num, ch_num, seq_len, patch_len)
           # y: (seq_num, )
           self.seq_num, self.ch_num, seq_len, patch_len = x.shape
   
           self.run_mode = args.run_mode
   
           self.x = x
           self.y = y
           self.f = np.abs(fft.fft(self.x)) #/(window_length) # rfft for real value inputs.
   
           if args.run_mode == "unsupervised":  # no need to apply Augmentations in other modes
               self.aug1_x = DataTransform_TD(self.x, args)
               self.aug1_f = DataTransform_FD(self.f, args) # [7360, 1, 90]
   
           self.nProcessLoader = args.n_process_loader
           self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)
   
       def __getitem__(self, index):
           if self.run_mode == "unsupervised":
               return self.x[index], self.aug1_x[index],  \
                      self.f[index], self.aug1_f[index]
           elif self.run_mode == 'finetune' or self.run_mode == 'test':
               return self.x[index], self.y[index], self.x[index], \
                      self.f[index], self.f[index]
           else:
               raise NotImplementedError(f'Undefined running mode {self.run_mode}')
   
       def __len__(self):
           return self.seq_num
   
       def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
           return DataLoader(self,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle)
   ```

   If your model is so trivial that it just need raw data `x` and labels `y` as the input of the `forward_propagate` function, you can directly use the class `DefaultDataset` in the `default_dataset.py`. Thus there's no need to write your own dataset class!

5. In the `BrainBenchmark/utils/meta_info.py`, 

   - Add a line `'NAME': NAME_Trainer,` to the dictionary `trainer_dict`, and import the class `NAME_Trainer` here. 
   - Add a line `'NAME': NAME,` to the dictionary `model_dict`, and import the class `NAME` here. 
   - Add a line `'NAME': NAME_Dataset, ` to the dictionary `dataset_class_dict`, and import the model class `NAME_Dataset` here. 

By the steps above, a new method can be added to the benchmark. 


# Benchmark Table
## Model
| Mode Name | paper | code |
| ---------- | ---------- | ---------- |
| BIOT | Biot: Biosignal transformer for cross-data learning in the wild | [BIOT](https://github.com/ycq091044/BIOT)|
| BrainBERT | Brainbert: Self-supervised representation learning for intracranial recordings | [Brainbert](https://github.com/czlwang/BrainBERT) |
| Brant1 | Brant: Foundation model for intracranial neural signal | [Brant](https://zju-brainnet.github.io/Brant.github.io/)
| Brant2 | BrainWave: A Brain Signal Foundation Model for Clinical Applications | - |
| GPT4TS | One Fits All:Power General Time Series Analysis by Pretrained LM | [GPT4TS](https://github.com/ekto42/GPT4TS) |
| LaBraM | Large brain model for learning generic representations with tremendous EEG data in BCI | [LaBraM](https://github.com/935963004/LaBraM) |
| SimMTM | SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling | [SimMTM](https://github.com/thuml/SimMTM) |
| TF-C | Self-Supervised Contrastive Pre-Training for Time  Series via Time-Frequency Consistency | [TF-C](https://github.com/mims-harvard/TF-C-pretraining) |

## Dataset
The benchmark contains 9 public datasets and 4 private datasets. 
* public datasets: [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [Mayo-Clinic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [Siena](https://www.mdpi.com/2227-9717/8/7/846), [HUSM](https://figshare.com/articles/dataset/EEG_Data_New/4244171), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [RepOD](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441), [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/)
* private datasets: SeizureA, SeizureB, SeizureC, Clinical

## Benchmark

| Mode Name | Dataset | Acc | Prec | Rec | F2 | AUPRC | AUROC |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Brant1 | MAYO | $90.53 \pm 2.59$ | $64.90 \pm 18.24$ | $74.12 \pm 22.09$ | $69.87 \pm 18.56$ | $70.05 \pm 15.17$ | $93.21 \pm 3.11$ |
| Brant1 | UCSD_ON | $71.51 \pm 26.04$ | $50.83 \pm 50.03$ | $60.00 \pm 54.77$ | $57.10 \pm 52.46$ | $70.19 \pm 27.65$ | $74.34 \pm 23.89$ |
| Brant1 | FNUSA | $82.13 \pm 8.59$ | $70.96 \pm 15.92$ | $75.25 \pm 17.44$ | $73.17 \pm 14.33$ | $80.86 \pm 12.85$ | $86.36 \pm 12.01$ |
| Brant1 | HUSM | $77.36 \pm 20.89$ | $65.73 \pm 39.04$ | $75.68 \pm 42.49$ | $73.23 \pm 41.45$ | $88.94 \pm 12.59$ | $87.14 \pm 15.42$ |
| Brant1 | RepOD | $71.01 \pm 23.54$ | $71.88 \pm 24.74$ | $98.28 \pm 2.41$ | $89.74 \pm 6.04$ | $77.31 \pm 24.15$ | $75.11 \pm 27.62$ |
| Brant1 | CHBMIT | $92.03 \pm 1.27$ | $41.30 \pm 48.36$ | $13.80 \pm 16.28$ | $15.84 \pm 18.58$ | $36.72 \pm 22.70$ | $71.14 \pm 17.65$ |
| Brant1 | Siena | $90.30 \pm 2.39$ | $44.85 \pm 36.08$ | $29.90 \pm 17.49$ | $31.19 \pm 17.96$ | $45.31 \pm 21.61$ | $86.17 \pm 5.51$ |
| Brant2 | MAYO | 91.393 | 73.369 | 66.125 | 66.69 | 73.302 | 
| Brant2 | UCSD |  |  |  |  |  | 
| Brant2 | FNUSA |  |  |  |  |  | 
| Brant2 | HUSM |  |  |  |  |  | 
| Brant2 | RepOD |  |  |  |  |  | 
| Brant2 | CHBMIT |  |  |  |  |  | 
| Brant2 | Siena |  |  |  |  |  | 
| BrainBERT | MAYO | $72.33 \pm 12.10$ | $37.60 \pm 22.43$ | $83.63 \pm 14.03$ | $61.32 \pm 12.13$ | $61.34 \pm 20.24$ | $88.52 \pm 6.70$ |
| BrainBERT | UCSD | $52.75 \pm 6.64$ | $50.31 \pm 35.71$ | $61.52 \pm 52.76$ |$51.90 \pm 43.32$ | $82.17 \pm 7.45$ | $78.10 \pm 15.95$ |
| BrainBERT | FNUSA | $50.03 \pm 13.24$ | $37.30 \pm 7.70$ | $88.32 \pm 12.41$ |$67.95 \pm 6.24$ | $68.91 \pm 8.13$ | $79.44 \pm 8.20$ |
| BrainBERT | HUSM | $71.56 \pm 6.13$ | $65.24 \pm 5.47$ | $99.74 \pm 0.58$ | $90.08 \pm 2.16$ | $89.73 \pm 4.16$ | $90.47 \pm 2.62$ |
| BrainBERT | RepOD | $90.52 \pm 80.22$ | $87.79 \pm 77.19$ | $98.42 \pm 99.51$ | $96.09 \pm 92.99$ | $99.41 \pm 95.22$ | $98.75 \pm 94.49$ |
| BrainBERT | CHBMIT | $86.62 \pm 84.33$ | $39.72 \pm 33.37$ | $89.20 \pm 70.36$ | $71.41 \pm 57.11$ | $66.85 \pm 55.19$ | $92.82 \pm 82.70$ |
| BrainBERT | Siena | $77.07 \pm 11.84$ | $26.45 \pm 11.45$ | $68.47 \pm 14.01$ | $49.69 \pm 10.55$ | $39.48 \pm 17.11$ | $80.89 \pm 9.18$ |
| GPT4TS | MAYO | $72.24 \pm 15.86$ | $37.47 \pm 20.97$ | $82.13 \pm 15.36$ | $61.13 \pm 13.94$ | $55.27 \pm 24.63$ | $86.10 \pm 9.46$ |
| GPT4TS | UCSD | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ | $100.00 \pm 0.00$ |
| GPT4TS | FNUSA | $48.20 \pm 14.81$ | $36.87 \pm 7.85$ | $91.39 \pm 10.63$ | $69.15 \pm 5.14$ | $64.10 \pm 6.68$ | $78.74 \pm 7.09$ |
| GPT4TS | HUSM | $70.42 \pm 2.76$ | $64.58 \pm 2.74$ | $99.78 \pm 0.49$ | $89.94 \pm 0.90$ | $90.73 \pm 2.63$ | $90.34 \pm 2.01$ | 
| GPT4TS | RepOD | $63.91 \pm 11.51$ | $61.43 \pm 8.73$ | $99.54 \pm 1.02$ | $88.23 \pm 3.26$ | $93.96 \pm 3.55$ | $92.35 \pm 4.27$ |
| GPT4TS | CHBMIT | $85.63 \pm 10.79$ | $41.42 \pm 17.28$ | $65.78 \pm 22.33$ | $56.14 \pm 19.04$ | $55.79 \pm 22.16$ | $81.69 \pm 12.29$ | 
| GPT4TS | Siena | $82.48 \pm 4.62$ | $31.55 \pm 7.88$ | $68.51 \pm 8.79$ | $55.01 \pm 7.84$ | $45.50 \pm 17.48$ | $84.13 \pm 4.30$ | 
| SimMTM | MAYO | 87.332 | 66.871 | 24.03 | 26.157 | 55.788 | 
| SimMTM | UCSD |  |  |  |  |  | 
| SimMTM | FNUSA |  |  |  |  |  | 
| SimMTM | HUSM |  |  |  |  |  | 
| SimMTM | RepOD |  |  |  |  |  | 
| SimMTM | CHBMIT |  |  |  |  |  | 
| SimMTM | Siena |  |  |  |  |  | 
| TF-C | MAYO | 88.635 | 54.12 | 48.162 | 48.276 | 48.975 | 
| TF-C | UCSD |  |  |  |  |  | 
| TF-C | FNUSA | 86.475 | 91.382 | 80.908 | 82.807 | 95.129 | 
| TF-C | HUSM |  |  |  |  |  | 
| TF-C | RepOD |  |  |  |  |  | 
| TF-C | CHBMIT |  |  |  |  |  | 
| TF-C | Siena |  |  |  |  |  | 
| BIOT | MAYO | $82.29 \pm 5.32$ | $44.70 \pm 23.58$ | $55.75 \pm 14.68$ | $50.33 \pm 11.29$ | $47.39 \pm 18.72$ | $78.68 \pm 7.23$ |
| BIOT | UCSD | $48.35 \pm 6.54$ | $49.12 \pm 4.95$ | $96.00 \pm 8.94$ | $80.59 \pm 7.51$ | $48.10 \pm 9.64$ | $42.82 \pm 10.51$ |
| BIOT | FNUSA | $57.94 \pm 10.62$ | $39.79 \pm 6.62$ | $74.83 \pm 12.77$ | $63.09 \pm 9.26$ | $55.35 \pm 15.99$ | $70.38 \pm 15.87$ |
| BIOT | HUSM | $46.46 \pm 7.05$ | $50.23 \pm 14.77$ | $62.15 \pm 48.11$ | $53.50 \pm 38.79$ | $61.03 \pm 10.77$ | $57.37 \pm 13.71$ |
| BIOT | RepOD | $48.01 \pm 16.48$ | $67.60 \pm 32.24$ | $45.09 \pm 40.52$ | $42.50 \pm 35.26$ | $60.29 \pm 20.04$ | $48.77 \pm 22.55$ |
| BIOT | CHBMIT | $87.14 \pm 2.61$ | $23.19 \pm 15.27$ | $16.38 \pm 15.36$ | $16.59 \pm 14.61$ | $19.64 \pm 11.34$ | $63.07 \pm 8.74$ |
| BIOT | Siena | $39.25 \pm 32.13$ | $11.37 \pm 5.34$ | $65.73 \pm 30.42$ | $28.96 \pm 5.38$ | $17.02 \pm 12.03$ | $54.82 \pm 12.50$ |
| LaBraM | MAYO | $88.44 \pm 3.72$ | $69.80 \pm 23.37$ | $34.65 \pm 11.64$ | $38.25 \pm 12.19$ | $60.31 \pm 22.64$ | $87.81 \pm 8.15$ |
| LaBraM | UCSD | $52.42 \pm 4.27$ | $37.89 \pm 52.02$ | $5.20 \pm 8.11$ | $6.24 \pm 9.66$ | $67.83 \pm 8.58$ | $67.12 \pm 10.63$ |
| LaBraM | FNUSA | $80.84 \pm 7.46$ | $76.36 \pm 13.03$ | $51.55 \pm 16.56$ | $54.82 \pm 16.44$ | $72.08 \pm 14.70$ | $81.01 \pm 10.23$ |
| LaBraM | HUSM | $67.59 \pm 13.20$ | $84.30 \pm 6.87$ | $49.03 \pm 32.49$ | $51.08 \pm 31.75$ | $81.71 \pm 3.89$ | $84.02 \pm 4.54$ |
| LaBraM | RepOD | $59.24 \pm 7.04$ | $67.26 \pm 11.69$ | $64.14 \pm 21.26$ | $62.56 \pm 12.55$ | $72.98 \pm 6.74$ | $71.24 \pm 3.25$ |
| LaBraM | CHBMIT | $92.17 \pm 1.72$ | $56.87 \pm 41.01$ | $17.43 \pm 20.20$ | $19.85 \pm 22.77$ | $57.21 \pm 19.49$ | $85.78 \pm 10.10$ |
| LaBraM | Siena | $90.82 \pm 0.41$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ | $33.56 \pm 15.89$ | $79.59 \pm 7.05$ |

| Mode Name | Dataset | TopKAcc | Sens | Spec | MF1 | Kappa|
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| BrainBERT | ISRUC | $64.20 \pm 3.93$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $4.71 \pm 0.25$ | $0.00 \pm 0.00$ |
| BrainBERT | SleepEDFx | $93.38 \pm 1.13$ | $88.37 \pm 0.70$ | $88.37 \pm 0.70$ | $45.91 \pm 3.34$ | $38.53 \pm 4.13$ |
| BIOT | ISRUC | $66.19 \pm 2.79$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $4.72 \pm 0.25$ | $0.00 \pm 0.00$ |
| BIOT | SleepEDFx | $89.22 \pm 0.65$ | $86.37 \pm 0.64$ | $86.37 \pm 0.64$ | $38.96 \pm 3.35$ | $30.68 \pm 3.70$ |
| LaBraM | ISRUC | $72.75 \pm 1.90$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $9.70 \pm 0.72$ | $0.00 \pm 0.00$ |
| LaBraM | SleepEDFx | $95.03 \pm 1.08$ | $90.88 \pm 0.59$ | $90.88 \pm 0.59$ | $53.84 \pm 2.04$ | $54.15 \pm 2.98$ |