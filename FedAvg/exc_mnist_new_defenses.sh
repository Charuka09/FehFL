#!/bin/bash
# Experiments: New SOTA defenses (FLTrust, DP-FL, BREA, RSA) under label-flipping attack
# Dataset: MNIST, 100 users (fix_total), 20 attackers, 200 epochs, non-IID

echo "=== New defense baselines: label-flipping attack, 20 attackers ==="

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg fltrust --root_dataset_size 100 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg brea --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg rsa --dataset mnist &

wait
echo "=== New defense baselines: label-flipping attack, 5 attackers ==="

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg fltrust --root_dataset_size 100 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg brea --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 5 --attacker_ep 5 --num_users 100 --attack_label 1 \
    --agg rsa --dataset mnist &

wait
echo "Done."
