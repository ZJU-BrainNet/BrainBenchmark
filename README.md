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
--load_ckpt_path '/data/brainnet/benchmark/ckpt/' 
--save_ckpt_path '/data/brainnet/benchmark/ckpt/' 
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
| TF-C | Self-Supervised Contrastive Pre-Training for Time  Series via Time-Frequency Consistency | [TF-C](https://github.com/mims-harvard/TF-C-pretraining)

## Dataset
The benchmark contains 9 public datasets and 4 private datasets. 
* public datasets: [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/), [Mayo-Clinic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [Siena](https://www.mdpi.com/2227-9717/8/7/846), [HUSM](https://figshare.com/articles/dataset/EEG_Data_New/4244171), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [RepOD](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441), [SleepEDFx](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/)
* private datasets: SeizureA, SeizureB, SeizureC, Clinical

## Benchmark

| Mode Name | Dataset | Acc | Prec | Rec | F2 | AUCROC | AUPRC |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
|   Brant1  |  MAYO | 91.318 | 71.602 | 67.493 | 67.887 | 72.223 | 
|   Brant1 | UCSD | 100 | 100 | 100 | 100 | 100 | 100 |
|   Brant1 | FNUSA | 91.1 | 84.81 | 90.07 | 88.85 | 96.63 | 93.42
|   Brant1 | HUSM | 93.28 | 92.45 | 94.71 | 94.25 | 97.96 | 97.99
|   Brant1 | RepOD | 63.31 | 66.68 | 88.1 | 81.06 | 87.71 | 82.25
|   Brant1 | SeizureA | 91.77 | 21 | 13.99 | 14.44 | 61.31 | 21.23
|   Brant1 | SeizureB | 79.92 | 46.8 | 72.89 | 62.53 | 92.59 | 62.7 
|   Brant1 | SeizureC | 51.73 | 0 | 0 | 0 | 59.75 | 57.53 
|   Brant1 | CHBMIT | 91.02 | 66.66 | 18.07 | 20.23 | 80.46 | 41.18 
|   Brant1 | Siena |  |  |  |  |  | 
|   Brant1 | Clinical |  |  |  |  |  | 
|   Brant2 | MAYO | 91.393 | 73.369 | 66.125 | 66.69 | 73.302 | 
|   Brant2 | UCSD |  |  |  |  |  | 
|   Brant2 | FNUSA |  |  |  |  |  | 
|   Brant2 | HUSM |  |  |  |  |  | 
|   Brant2 | RepOD |  |  |  |  |  | 
|   Brant2 | CHBMIT |  |  |  |  |  | 
|   Brant2 | Siena |  |  |  |  |  | 
|   Brant2 | Clinical | 86.085 | 45.014 | 39.116 | 39.64 | 41.625 | 
|   BrainBERT | MAYO | 83.077 | 58.197 | 75.845 | 67.987 | 69.289 | 
|   BrainBERT | UCSD |  |  |  |  |  | 
|   BrainBERT | FNUSA | 84.473 | 79.49 | 67.787 | 69.668 | 80.961 | 
BrainBERT | HUSM |  |  |  |  |  | 
BrainBERT | RepOD |  |  |  |  |  |  
BrainBERT | CHBMIT |  |  |  |  |  | 
BrainBERT | Siena |  |  |  |  |  | 
BrainBERT | Clinical | 78.636 | 49.551 | 67.252 | 61.133 | 62.824 | 
GPT4TS | MAYO | 70.036 | 34.094 | 74.416 | 59.864 | 61.435 | 
GPT4TS | UCSD |  |  |  |  |  | 
GPT4TS | FNUSA | 46.175 | 46.175 | 100 | 78.466 | 40.822 | 
GPT4TS | HUSM |  |  |  |  |  | 
GPT4TS | RepOD |  |  |  |  |  | 
GPT4TS | CHBMIT |  |  |  |  |  | 
GPT4TS | Siena |  |  |  |  |  | 
GPT4TS | Clinical | 67.345 | 50.211 | 79.753 | 66.836 | 55.212 | 
SimMTM | MAYO | 87.332 | 66.871 | 24.03 | 26.157 | 55.788 | 
SimMTM | UCSD |  |  |  |  |  | 
SimMTM | FNUSA |  |  |  |  |  | 
SimMTM | HUSM |  |  |  |  |  | 
SimMTM | RepOD |  |  |  |  |  | 
SimMTM | CHBMIT |  |  |  |  |  | 
SimMTM | Siena |  |  |  |  |  | 
TF-C | MAYO | 88.635 | 54.12 | 48.162 | 48.276 | 48.975 | 
TF-C | UCSD |  |  |  |  |  | 
TF-C | FNUSA | 86.475 | 91.382 | 80.908 | 82.807 | 95.129 | 
TF-C | HUSM |  |  |  |  |  | 
TF-C | RepOD |  |  |  |  |  | 
TF-C | CHBMIT |  |  |  |  |  | 
TF-C | Siena |  |  |  |  |  | 
TF-C | Clinical | 69.774 | 36.014 | 56.045 | 47.266 | 51.283 | 
BIOT | MAYO | 88.952 | 71.129 | 73.969 | 72.494 | 82.224 | 
BIOT | UCSD | 59.899 | 37.5 | 46.552 | 42.996 | 60.464 | 
BIOT | FNUSA | 67.586 | 53.405 | 85.839 | 74.469 | 84.384 | 
BIOT | HUSM | 48 | 48.877 | 67.019 | 60.187 | 44.032 | 
BIOT | RepOD | 60.709 | 67.423 | 79.245 | 74.395 | 54.206 | 
BIOT | SeizureA | 85.688 | 50.063 | 75.283 | 61.412 | 89.962 | 
BIOT | SeizureB | 54.67 | 49.046 | 45.354 | 39.93 | 64.557 | 
BIOT | SeizureC | 53.37 | 45.088 | 83.897 | 58.681 | 52.197 | 
BIOT | CHBMIT | 75.429 | 17.364 | 25.311 | 19.631 | 58.741 | 
BIOT | Siena | 65.169 | 12.857 | 29.73 | 13.468 | 57.135 | 
BIOT | Clinical |  |  |  |  |  | 
LaBraM | MAYO |  |  |  |  |  | 
LaBraM | UCSD |  |  |  |  |  | 
LaBraM | FNUSA | 73.482 | 60.211 | 82.579 | 76.318 | 82.621 | 
LaBraM | HUSM |  |  |  |  |  | 
LaBraM | RepOD |  |  |  |  |  | 
LaBraM | SeizureA | 88.28 | 38.958 | 65.208 | 55.625 | 88.894 | 
LaBraM | SeizureB | 54.241 | 49.66 | 60.768 | 47.913 | 69.352 | 
LaBraM | SeizureC | 51.622 | 44.634 | 95.909 | 65.637 | 60.898 | 
LaBraM | CHBMIT | 91.931 | 53.317 | 71.625 | 67.013 | 91.11 | 
LaBraM | Siena | 84.971 | 35.782 | 62.705 | 53.893 | 84.87 | 