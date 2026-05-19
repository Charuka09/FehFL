#!/bin/bash
# Experiments: Advanced attacks (model poisoning, min-max) vs all defenses
# Dataset: CIFAR-100, 50 users (fix_total), 20 attackers, 500 epochs, iid

echo "=== Model poisoning attack: CIFAR-100, 20 attackers ==="

for AGG in euclidean_distance average median trimmed_mean krum brea rsa; do
    python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
        --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label -1 \
        --attack_type model_poisoning --mp_boost 10.0 \
        --agg $AGG --noise 0 --dataset cifar-100 &
done
wait

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg fltrust --root_dataset_size 100 --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --noise 0 --dataset cifar-100 &

wait
echo "=== Min-max attack: CIFAR-100, 20 attackers ==="

for AGG in euclidean_distance average median trimmed_mean krum brea rsa; do
    python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
        --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label -1 \
        --attack_type min_max --mp_boost 2.0 \
        --agg $AGG --noise 0 --dataset cifar-100 &
done
wait

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg fltrust --root_dataset_size 100 --noise 0 --dataset cifar-100 &

python main_nn.py --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --noise 0 --dataset cifar-100 &

wait
echo "Done."