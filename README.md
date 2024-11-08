# Recursive Euclidean Distance-based Robust Aggregation Technique for Federated Learning

## Introduction


This is a PyTorch implementation of our [paper](https://arxiv.org/abs/2303.11337).  * This repository used code from [federated learning](https://github.com/shaoxiongji/federated-learning) and [Attack-Resistant-Federated-Learning](https://github.com/fushuhao6/Attack-Resistant-Federated-Learning)

## Citing Recursive Euclidean Distance Based Robust Aggregation Technique For Federated Learnin
``
@article{recursiveEuclideanDistanc,
    title={Recursive Euclidean Distance Based Robust Aggregation Technique For Federated Learning},
    author={Charuka Herath, Yogachandran Rahulamathavan, Xiaolan Liu},
    journal={https://arxiv.org/abs/2303.11337},
    year={2023}
}

@article{FheFL,
    title={FheFL: Fully Homomorphic Encryption Friendly Privacy-Preserving Federated Learning with Byzantine Users},
    author={Yogachandran Rahulamathavn, Charuka Herath, Xiaolan Liu, Sangarapillai Lambotharan, Carsten Maple},
    journal={https://arxiv.org/abs/2306.05112},
    year={2023}
}
```

## Requirements: Software

1. Pytorch from [the offical repository](https://pytorch.org/).
2. Install tensorboardX.
```
pip install tensorboardX
```
## Preparation for Training & Testing
1. The code will automatically download MNIST dataset.

## Usage
### Label-flipping attack experiments
Label Flipping attack on MNIST
```
python main_nn.py --model smallcnn --epochs 100 --gpu 0 --iid 0 --fix_total --frac 0.1 --num_attackers 4 --attacker_ep 10 --num_users 100 --attack_label 1 --agg euclidean_distance --noise 1
```
Change `--agg` tag to select aggregation algorithm and change `--num_attackers` to specify the number of attackers.

### Run based on Dynamic data distribution setting
```
python main_nn.py --model smallcnn --epochs 100 --gpu 0 --iid 0 --is_dynamic 1  --fix_total --frac 0.1 --num_attackers 0  --attacker_ep 0 --num_users 100 --attack_label 1 --agg average --is_dynamic 1
```
### LSFE approach with noise added secure aggregation
#### Label flipping attack --> 10 attackers and 5 attacker epochs
```
python main_nn.py --model smallcnn --epochs 1 --gpu 0 --iid 0 --fix_total --frac 0.1 --num_attackers 10 --attacker_ep 5 --num_users 100 --attack_label 1 --agg euclidean_distance --noise 1
```

#### Backdoor poisoning attack
```
python main_nn.py --model smallcnn --epochs 100 --gpu 0 --iid 0 --fix_total --frac 0.1 --num_attackers 10 --attacker_ep 5 --num_users 100 --attack_label 1 --agg euclidean_distance --noise 1 --is_backdoor True
```

### LSFE approach with added noise

```
python main_nn.py --model smallcnn --epochs 1 --gpu 0 --iid 0 --fix_total --frac 0.1 --num_attackers 10 --attacker_ep 5 --num_users 100 --attack_label 1 --agg lsfe --noise 0
```


## LSFE scripts

### mnist
```
python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg euclidean_distance --noise 0 --dataset mnist

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg average --noise 0 --dataset mnist

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg median --noise 0 --dataset mnist

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg trimmed_mean --noise 0 --dataset mnist

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg krum --noise 0 --dataset mnist

```

### CIFAR-10
```
python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 3 --agg euclidean_distance --noise 0 --dataset cifar

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 3 --agg average --noise 0 --dataset cifar

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 3 --agg median --noise 0 --dataset cifar

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 3 --agg trimmed_mean --noise 0 --dataset cifar

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 3 --agg krum --noise 0 --dataset cifar

```

### CIFAR-100
```
python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 3 --agg euclidean_distance --noise 0 --dataset cifar-100

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 3 --agg average --noise 0 --dataset cifar-100

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 3 --agg median --noise 0 --dataset cifar-100

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 3 --agg trimmed_mean --noise 0 --dataset cifar-100

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 3 --agg krum --noise 0 --dataset cifar-100

```


