# From Stagnation to Elevation: Empower Aggregation in Federated Learning through the Lens of Sparsity


This is the offical implementation for the submission titled From Stagnation to Elevation: Empower Aggregation in Federated Learning through the Lens of Sparsity.

## Requirements
- torch==1.12.1
- torchvision==0.13.1
- numpy>=1.18.0
- easydict==1.9'
- yapf==0.29.0
- timm
- click
- prettytable
- portalocker
- einops
- scipy
- six
- tqdm
- matplotlib


## Dataset
Please run the scripts in data folder to download and prepare the dataset before training. (For CIFAR10/100 datasets, they are downloaded automatically)


## Training

To train and evaluate models for **CIFAR-10**, run this command: 

```
noniid='dirichlet'
model='vgg11_bn'
client_num=100
alpha=0.1
train_num=100
python -u ./flzoo/cifar10/cifar10_fedlips_resnet_config.py --dst rigl --model $model --prune_ll large --prune_mode global_sel --regrow_method sensityk --noniid $noniid --client_num $client_num --train_num $train_num --seed 2 --density_init 1.0 --alpha $alpha --no_margin True --update_d 0.5 --update_freq 5

```

To train and evaluate models for **CIFAR-100**, run this command: 

```
noniid='dirichlet'
model='ResNet8'
client_num=100
alpha=0.1
train_num=100
python -u ./flzoo/cifar100/cifar100_fedlips_resnet_config.py --dst rigl --model $model --prune_ll large --prune_mode global_sel --regrow_method sensityk --noniid $noniid --client_num $client_num --train_num $train_num --seed 2 --density_init 1.0 --alpha $alpha --no_margin True --update_d 0.3 --update_freq 5

```



## Acknowledgements
We appreciate the following github repos a lot for their valuable code.

https://github.com/FLAIR-Community/Fling
​        
​    
