#!/bin/bash
# Experiments: New SOTA defenses (FLTrust, DP-FL, BREA, RSA) under label-flipping attack
# Dataset: CIFAR-10, 50 users (fix_total), 20 / 5 attackers, 200 epochs, iid

echo "=== New defense baselines: CIFAR-10, label-flipping, 20 attackers ==="

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg fltrust --root_dataset_size 100 --noise 0 --dataset cifar &

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --noise 0 --dataset cifar &

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg brea --noise 0 --dataset cifar &

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg rsa --noise 0 --dataset cifar &

wait
echo "=== New defense baselines: CIFAR-10, label-flipping, 5 attackers ==="

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg fltrust --root_dataset_size 100 --noise 0 --dataset cifar &

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --noise 0 --dataset cifar &

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg brea --noise 0 --dataset cifar &

python main_nn.py --epochs 200 --gpu 1 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 50 --attack_label 3 \
    --agg rsa --noise 0 --dataset cifar &

wait
echo "Done."
