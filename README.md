# TransferLearning Experiments on Fashion Dataset

Dependencies:
1. Pytorch : 1.0.1post2
2. Python :  3.6.9
3. Torchvision : 0.2.2

## Pre-Training:

```Shell
optional arguments:
  --lr                 learning rate for pretraining
  --batch_size         num_batch
  --epochs             training epochs 
  --add_sampler        1 if weighted sampler
  --dir_               image_dir where the images are present
  --resize             model input size [224, 224, 2(for Random resize)] 
  --exp_name           Name of the experiment
  --schedule           lr drop schedules [60, 80, 120]
  
```
Pre-training ResNet18:

```
python main.py --lr 0.1 --dir_ IMG_DIR --batchsize 128 --resize 224 224 2 --exp_name pretrain_best --schedule 80 60 120 --epochs 200 

```

## Finetuning:

```Shell
Additional optional arguments:
  --finetune_lr        learning rate for finetuning
  --batch_size         num_batch
  --epochs             training epochs 
  --add_sampler        1 if weighted sampler
  --dir_               image_dir where the images are present
  --finetune_freeze    True if freeze the layers
  --freeze_layers      [fc., layer4.] which layers to train
  --train_between      To train whole network in between
  --switch_all         unfreeze all layer at epoch - 40
  
  --resize             model input size [224, 224, 2(for Random resize)] 
  --exp_name           Name of the experiment
  --schedule           lr drop schedules [60, 80, 120]
  
```

Finetuning pretrained  model

```
python main.py --finetune_lr 0.1 --dir_ IMG_DIR --batch_size 128 --resize 224 224 2  --model_weights PRETRAIN_MODEL_WEIGHTS --exp_name finetune_best --schedule  60 80 120 --epochs 200 --train_between  --switch_all 40 --freeze_layers fc. layer4. --finetune_freeze --schedule 60 80 120

```

Testing the pretrain/finetune models

```
python main.py --test pretrain_test/finetune_test --model_weights PRETRAIN/FINETUNE model  --dir_ IMG_DIR 

# outputs are stored as model_name.json

```





## Results

- Results on Fashion Pretrain:

| Model Name | Acc | Very Good  |  Good  | Medium  | Less  | Top 5 |
| --- | --- | --- | --- | --- |--- | --- |
| ResNet_80px  | 84.3 | 91.11 |82.23 | 89.03 | 68.52| 93.43
| CResNet_80px | 87.25|95.23| 84.29 | 91.21 | 70.2|96.72
| ResNet_IN_224px | 87.02|  95.16| 82.1 | 90.2 | 71.8|95.58
| ResNet_50 | 86.8 | 94.71 | 81.74 | 89.1| 72.31|95.21

- Results on Fashion Finetune:

| Model Name | Acc | 
| --- | --- | --- | --- | --- |--- | --- |
| ResNet_224px with freeze till layer 4 pretrained*  | 46.72 | 
| ResNet_224px with freeze only FC pretrained| 39.39|
| ResNet_224px with  weight decay 0  |37.09|   
| ResNet_224px trained from scratch | 45.82 | 
| ResNet_224px with weighted Sampler | 12-11 |

