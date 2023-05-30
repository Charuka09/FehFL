# Recursive Euclidean Distance-based Robust Aggregation Technique for Federated Learning

## Introduction


This is a PyTorch implementation of our [paper](https://arxiv.org/abs/2303.11337).  * This repository used code from [federated learning](https://github.com/shaoxiongji/federated-learning).

## Citing Attack-Resistant Federated Learning
```
@article{fu2019attackresistant,
    title={Recursive Euclidean Distance Based Robust Aggregation Technique For Federated Learning},
    author={Charuka Herath, Yogachandran Rahulamathavan, Xiaolan Liu},
    journal={https://arxiv.org/abs/2303.11337},
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
python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.1 --num_attackers 4 --attacker_ep 10 --num_users 100 --attack_label 1 --agg euclidean_distance
```
Change `--agg` tag to select aggregation algorithm and change `--num_attackers` to specify the number of attackers.