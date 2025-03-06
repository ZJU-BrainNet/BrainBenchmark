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

## Data Segmentation 

For each dataset you want to run experiments on, the first thing to do is to split the data into specific set on your device. Just run the `data_gene.py`, passing in the specific argument value: `dataset`, `sample_seq_num`, `seq_len`, and  `patch_len`.

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
--load_ckpt_path f'/data/brainnet/benchmark/ckpt' 
--save_ckpt_path f'/data/brainnet/benchmark/ckpt' 
```



# How to extend

## To add a new dataset

1. Split all the subjects in the new dataset into several groups (4-6 groups are recommended). Each group of data should be generated as a signle file named like `group_0_data.npy`, ..., `group_5_data.npy`. In each file, the shape of the numpy array is: `(seq_num, ch_num, seq_len, patch_len)` 

   The cooresponding label files should be named in similar format: `group_0_label.npy`, ..., `group_5_label.npy`. 

2. Then, add a new element in the `data_info_dict` from  `BrainBenchmark/data_process/data_info.py`. Taking MAYO as an example: 

   ```python
   'MAYO': {'data_path': '/MAYO_data_path/group_data/',
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
| BrainWave | BrainWave: A Brain Signal Foundation Model for Clinical Applications | - |
| GPT4TS | One Fits All:Power General Time Series Analysis by Pretrained LM | [GPT4TS](https://github.com/ekto42/GPT4TS) |
| LaBraM | Large brain model for learning generic representations with tremendous EEG data in BCI | [LaBraM](https://github.com/935963004/LaBraM) |
| SimMTM | SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling | [SimMTM](https://github.com/thuml/SimMTM) |
| TF-C | Self-Supervised Contrastive Pre-Training for Time  Series via Time-Frequency Consistency | [TF-C](https://github.com/mims-harvard/TF-C-pretraining) |

The models SimMTM and TF-C in the above model do not provide corresponding checkpoints, so we finetune them after pretraining on the matching dataset before testing.

## Dataset
The benchmark contains 9 public datasets and 4 private datasets. 
* public datasets: [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [Mayo-Clinic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [Siena](https://www.mdpi.com/2227-9717/8/7/846), [HUSM](https://figshare.com/articles/dataset/EEG_Data_New/4244171), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [RepOD](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441), [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/)
* private datasets: SeizureA, SeizureB, SeizureC, Clinical

## Benchmark
Currently, the benchmark is being reorganized. More models and datasets will be added in the future.
| Mode Name | Dataset | Acc | Prec | Rec | F2 | AUPRC | AUROC |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Brant1 | MAYO | $90.53 \pm 2.59$ | $64.90 \pm 18.24$ | $74.12 \pm 22.09$ | $69.87 \pm 18.56$ | $70.05 \pm 15.17$ | $93.21 \pm 3.11$ |
| Brant1 | FNUSA | $82.13 \pm 8.59$ | $70.96 \pm 15.92$ | $75.25 \pm 17.44$ | $73.17 \pm 14.33$ | $80.86 \pm 12.85$ | $86.36 \pm 12.01$ |
| Brant1 | CHBMIT | $92.03 \pm 1.27$ | $41.30 \pm 48.36$ | $13.80 \pm 16.28$ | $15.84 \pm 18.58$ | $36.72 \pm 22.70$ | $71.14 \pm 17.65$ |
| Brant1 | Siena | $90.30 \pm 2.39$ | $44.85 \pm 36.08$ | $29.90 \pm 17.49$ | $31.19 \pm 17.96$ | $45.31 \pm 21.61$ | $86.17 \pm 5.51$ |
| BrainBERT | MAYO | $72.33 \pm 12.10$ | $37.60 \pm 22.43$ | $83.63 \pm 14.03$ | $61.32 \pm 12.13$ | $61.34 \pm 20.24$ | $88.52 \pm 6.70$ |
| BrainBERT | FNUSA | $50.03 \pm 13.24$ | $37.30 \pm 7.70$ | $88.32 \pm 12.41$ |$67.95 \pm 6.24$ | $68.91 \pm 8.13$ | $79.44 \pm 8.20$ |
| BrainBERT | CHBMIT | $84.33 \pm 2.92$ | $33.37 \pm 4.80$ | $70.36 \pm 17.78$ | $57.11 \pm 11.94$ | $55.19 \pm 15.02$ | $82.70 \pm 9.62$ |
| BrainBERT | Siena | $77.07 \pm 11.84$ | $26.45 \pm 11.45$ | $68.47 \pm 14.01$ | $49.69 \pm 10.55$ | $39.48 \pm 17.11$ | $80.89 \pm 9.18$ |
| GPT4TS | MAYO | $72.24 \pm 15.86$ | $37.47 \pm 20.97$ | $82.13 \pm 15.36$ | $61.13 \pm 13.94$ | $55.27 \pm 24.63$ | $86.10 \pm 9.46$ |
| GPT4TS | FNUSA | $48.20 \pm 14.81$ | $36.87 \pm 7.85$ | $91.39 \pm 10.63$ | $69.15 \pm 5.14$ | $64.10 \pm 6.68$ | $78.74 \pm 7.09$ |
| GPT4TS | CHBMIT | $85.63 \pm 10.79$ | $41.42 \pm 17.28$ | $65.78 \pm 22.33$ | $56.14 \pm 19.04$ | $55.79 \pm 22.16$ | $81.69 \pm 12.29$ |
| GPT4TS | Siena | $82.48 \pm 4.62$ | $31.55 \pm 7.88$ | $68.51 \pm 8.79$ | $55.01 \pm 7.84$ | $45.50 \pm 17.48$ | $84.13 \pm 4.30$ |
| SimMTM | MAYO | $88.44 \pm 4.96$ | $84.62 \pm 11.53$ | $34.07 \pm 7.69$ | $38.47 \pm 7.98$ | $74.71 \pm 13.67$ | $93.62 \pm 3.35$ |
| SimMTM | FNUSA | $85.71 \pm 4.49$ | $80.76 \pm 16.95$ | $71.09 \pm 8.45$ | $71.83 \pm 4.48$ | $84.96 \pm 7.76$ | $91.78 \pm 4.01$ |
| SimMTM | CHBMIT | $91.32 \pm 2.81$ | $61.34 \pm 23.37$ | $36.05 \pm 17.95$ | $37.69 \pm 16.63$ | $43.05 \pm 16.07$ | $72.99 \pm 12.46$ |
| SimMTM | Siena | $90.81 \pm 4.85$ | $71.62 \pm 28.11$ | $40.83 \pm 16.77$ | $41.60 \pm 13.63$ | $55.06 \pm 14.37$ | $80.00 \pm 9.89$ |
| TF-C | MAYO | $87.67 \pm 1.75$ | $57.82 \pm 20.36$ | $48.34 \pm 8.81$ | $48.83 \pm 6.92$ | $55.29 \pm 16.02$ | $85.14 \pm 4.18$ |
| TF-C | FNUSA | $79.61 \pm 5.43$ | $67.35 \pm 9.34$ | $62.30 \pm 13.96$ | $62.77 \pm 12.39$ | $72.46 \pm 9.82$ | $83.34 \pm 7.29$ |
| TF-C | CHBMIT | $91.80 \pm 1.04$ | $65.95 \pm 16.21$ | $24.33 \pm 11.28$ | $27.44 \pm 12.40$ | $43.89 \pm 14.56$ | $78.72 \pm 11.67$ |
| TF-C | Siena | $92.55 \pm 2.14$ | $60.32 \pm 17.54$ | $41.42 \pm 21.82$ | $43.80 \pm 21.60$ | $56.09 \pm 18.52$ | $86.71 \pm 4.72$ |
| BIOT | MAYO | $82.29 \pm 5.32$ | $44.70 \pm 23.58$ | $55.75 \pm 14.68$ | $50.33 \pm 11.29$ | $47.39 \pm 18.72$ | $78.68 \pm 7.23$ |
| BIOT | FNUSA | $57.94 \pm 10.62$ | $39.79 \pm 6.62$ | $74.83 \pm 12.77$ | $63.09 \pm 9.26$ | $55.35 \pm 15.99$ | $70.38 \pm 15.87$ |
| BIOT | CHBMIT | $87.14 \pm 2.61$ | $23.19 \pm 15.27$ | $16.38 \pm 15.36$ | $16.59 \pm 14.61$ | $19.64 \pm 11.34$ | $63.07 \pm 8.74$ |
| BIOT | Siena | $39.25 \pm 32.13$ | $11.37 \pm 5.34$ | $65.73 \pm 30.42$ | $28.96 \pm 5.38$ | $17.02 \pm 12.03$ | $54.82 \pm 12.50$ |
| LaBraM | MAYO | $88.44 \pm 3.72$ | $69.80 \pm 23.37$ | $34.65 \pm 11.64$ | $38.25 \pm 12.19$ | $60.31 \pm 22.64$ | $87.81 \pm 8.15$ |
| LaBraM | FNUSA | $80.84 \pm 7.46$ | $76.36 \pm 13.03$ | $51.55 \pm 16.56$ | $54.82 \pm 16.44$ | $72.08 \pm 14.70$ | $81.01 \pm 10.23$ |
| LaBraM | CHBMIT | $92.17 \pm 1.72$ | $56.87 \pm 41.01$ | $17.43 \pm 20.20$ | $19.85 \pm 22.77$ | $57.21 \pm 19.49$ | $85.78 \pm 10.10$ |

| Mode Name | Dataset | TopKAcc | Sens | Spec | MF1 | Kappa|
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| BrainBERT | ISRUC | $64.20 \pm 3.93$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $4.71 \pm 0.25$ | $0.00 \pm 0.00$ |
| BrainBERT | SleepEDFx | $93.38 \pm 1.13$ | $88.37 \pm 0.70$ | $88.37 \pm 0.70$ | $45.91 \pm 3.34$ | $38.53 \pm 4.13$ |
| BIOT | ISRUC | $66.19 \pm 2.79$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $4.72 \pm 0.25$ | $0.00 \pm 0.00$ |
| BIOT | SleepEDFx | $89.22 \pm 0.65$ | $86.37 \pm 0.64$ | $86.37 \pm 0.64$ | $38.96 \pm 3.35$ | $30.68 \pm 3.70$ |
| LaBraM | ISRUC | $72.75 \pm 1.90$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $9.70 \pm 0.72$ | $0.00 \pm 0.00$ |
| LaBraM | SleepEDFx | $95.03 \pm 1.08$ | $90.88 \pm 0.59$ | $90.88 \pm 0.59$ | $53.84 \pm 2.04$ | $54.15 \pm 2.98$ |
| GPT4TS | ISRUC | $63.79 \pm 3.15$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $4.73 \pm 0.32$ | $0.00 \pm 0.00$ |
| GPT4TS | SleepEDFx | $92.96 \pm 1.81$ | $88.74 \pm 1.47$ | $88.74 \pm 1.47$ | $46.50 \pm 5.68$ | $40.25 \pm 8.34$ |
| SimMTM | ISRUC | $82.89 \pm 14.49$ | $85.98 \pm 5.28$ | $85.98 \pm 5.28$ | $37.36 \pm 24.87$ | $29.68 \pm 26.03$ |
| SimMTM | SleepEDFx | $97.19 \pm 0.72$ | $93.72 \pm 0.58$ | $93.72 \pm 0.58$ | $67.24 \pm 3.27$ | $67.12 \pm 3.08$ |
| TF-C | ISRUC | $72.88 \pm 2.01$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $9.67 \pm 0.73$ | $0.00 \pm 0.00$ |
| TF-C | SleepEDFx | $95.64 \pm 0.82$ | $91.91 \pm 0.70$ | $91.91 \pm 0.70$ | $60.19 \pm 2.41$ | $58.28 \pm 3.22$ |
| Brant1 | ISRUC | $72.00 \pm 4.06$ | $80.00 \pm 0.00$ | $80.00 \pm 0.00$ | $9.74 \pm 0.67$ | $0.00 \pm 0.00$ |
| Brant1 | SleepEDFx | $96.70 \pm 1.31$ | $92.87 \pm 1.24$ | $92.87 \pm 1.24$ | $61.32 \pm 6.55$ | $63.30 \pm 6.04$ |
