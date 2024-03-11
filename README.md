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

## Self-/unsupervised training

To perform the self-supervised or unsupervised training, please run the `b_train.py`. 
```
python b_train.py   --exp_id -1 \
                    --gpu_id 0 \
                    --train_mode unsupervised \
                    --dataset MAYO \
                    --sample_seq_num 2400 \
                    --seq_len 16 \
                    --patch_len 500 \
                    --data_load_dir '/data/brainnet/benchmark/gene_data/' \
                    --model TFC \ 
                    --cnn_in_channels 10 \
                    --cnn_kernel_size 8 \
                    --final_dim 512
```
Note that if you want to evaluate a result and make a direct performance comparison with other models, the following arguments about input data must be set according to the unified setting. These arguments includes `sample_seq_num`, `seq_len`, `patch_len`. 

Since many methods can only handle single-channel data, an inter-channel CNN is used to aggregate the representations across all channels to obtain a "subject-level" representation. The model architecture of this CNN can be set with the following arguments: `cnn_in_channels`, `cnn_kernel_size`. 

If you need to load ckpt (continue training from the last breakpoint), please add the `load_ckpt_path` argument (`None` if train from scratch). The path to save model checkpoints can also be set with the `save_ckpt_path` argument. 
```
--load_ckpt_path '/data/brainnet/benchmark/ckpt/' 
--save_ckpt_path '/data/brainnet/benchmark/ckpt/' 
```


# How to extend

## To add a new dataset

## To add a new method

Assume that the method name is `NAME`, 

1. Make a new directory `BrainBenchmark/model/NAME/`. 

2. Make a new file named `NAME.py` here, and write two functions in this file: `set_NAME_config` and `forward_NAME`. 

   In the function `set_NAME_config`, include all of the model parameters as input arguments, such that any user can set these arguments. For example: 

   ```python
   def set_TFC_config(parser):
       group_model = parser.add_argument_group('Model')
       group_model.add_argument('--Context_Cont_temperature', type=float, default=0.2)
       group_model.add_argument('--Context_Cont_use_cosine_similarity', action='store_false')
       group_model.add_argument('--augmentation_max_seg', type=int, default=12)
       group_model.add_argument('--augmentation_jitter_ratio', type=float, default=2)
       group_model.add_argument('--augmentation_jitter_scale_ratio', type=float, default=1.5)
       args = parser.parse_args()
       return args
   ```

   In the function `forward_NAME`, based on the data batch `data_packet` (this is determined by the `NAME_dataset` you write later), write the code for model forward propagation and loss calculation. If the code is different between the self-/unsupervision phase and fine-tuning phase, you can use the argument `args.train_mode`  to branch. For example:

   ```python
   def forward_TFC(args, data_packet, model, clsf, loss_func):
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

3. Then add any other files about your model in the directory `BrainBenchmark/model/NAME/`, such as the model class `NAME` and other utils. 

4. Make a new file `BrainBenchmark/datasets/NAME_dataset.py`, and write your dataset class `NAME_Dataset` here. Please make sure that the data composition returned in the `__getitem__` function matches what you receive in the `forward_NAME` function. Make sure that your class contains the following basic member functions: `__len__`, `get_data_loader`. For example: 

   ```python
   class TFC_Dataset(Dataset):
       def __init__(self, args, x, y):
           # x: (seq_num, ch_num, seq_len, patch_len)
           # y: (seq_num, )
           seq_num, ch_num, seq_len, patch_len = x.shape
           self.seq_num = seq_num
   
           self.train_mode = args.train_mode
           self.n_class = len(np.unique(y))
   				
           # code to calculate self.f, self.aug1_x, self.aug1_f
   
       def __getitem__(self, index):
           if self.train_mode == "unsupervised":
               return self.x[index], self.aug1_x[index],  \
                      self.f[index], self.aug1_f[index]
           elif self.train_mode == 'finetune':
               return self.x[index], self.y[index], self.x[index], \
                      self.f[index], self.f[index]
           else:
               raise NotImplementedError(f'Undefined training mode {self.train_mode}')
   
       def __len__(self):
           return self.seq_num
   
       def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
           return DataLoader(self,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle)
   ```

5. In the `BrainBenchmark/utils/meta_info.py`, 

   - Add a line `'NAME': set_NAME_config,` to the dictionary `set_model_config_dict`, and import the function `set_NAME_config` here. 
   - Add a line `'NAME': forward_NAME,` to the dictionary `forward_dict`, and import the function `forward_NAME` here. 
   - Add a line `'NAME': NAME, ` to the dictionary `model_dict`, and import the model class `NAME` here. 
   - Add a line `'NAME': NAME_dataset,` to the dictionary `dataset_class_dict`, and import the dataset class `NAME_dataset` here. 
   - Add a line `'NAME': LOSS_FUNC,` to the dictionary `loss_func_dict`, where `LOSS_FUNC` is the loss function your model uses. 

By the steps above, a new method can be added to the benchmark. 



