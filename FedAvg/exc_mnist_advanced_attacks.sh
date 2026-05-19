#!/bin/bash
# Experiments: Advanced attacks (model poisoning, min-max) vs all defenses
# Dataset: MNIST, 100 users (fix_total), 20 attackers, 200 epochs, non-IID

echo "=== Model poisoning attack vs all defenses, 20 attackers ==="

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg euclidean_distance --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg average --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg median --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg trimmed_mean --dataset mnist &

wait

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg krum --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg fltrust --root_dataset_size 100 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg brea --dataset mnist &

wait

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type model_poisoning --mp_boost 10.0 \
    --agg rsa --dataset mnist &

wait
echo "=== Min-max attack vs all defenses, 20 attackers ==="

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg euclidean_distance --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg average --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg median --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg trimmed_mean --dataset mnist &

wait

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg krum --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg fltrust --root_dataset_size 100 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg dp_fl --dp_clip 1.0 --dp_sigma 0.1 --dataset mnist &

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg brea --dataset mnist &

wait

python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 0 --fix_total --frac 0.2 \
    --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label -1 \
    --attack_type min_max --mp_boost 2.0 \
    --agg rsa --dataset mnist &

wait
echo "Done."
