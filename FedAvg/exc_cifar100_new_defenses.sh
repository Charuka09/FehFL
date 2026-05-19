#!/bin/bash
# Experiments: New SOTA defenses (FLTrust, DP-FL, BREA, RSA) under label-flipping attack
# Dataset: CIFAR-100, 50 users (fix_total), 20 / 10 attackers, 500 epochs, iid

echo "=== New defense baselines: CIFAR-100, label-flipping, 20 attackers ==="

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg fltrust --root_dataset_size 100 --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg brea --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg rsa --noise 0 --dataset cifar-100 &

wait
echo "=== New defense baselines: CIFAR-100, label-flipping, 10 attackers ==="

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 10 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg fltrust --root_dataset_size 100 --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 10 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 10 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg brea --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 10 --attacker_ep 5 --num_users 50 --attack_label 10 \
    --agg rsa --noise 0 --dataset cifar-100 &

wait
echo "Done."
