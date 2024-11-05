#!/bin/bash
# Label flipping
# python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --is_backdoor True --agg euclidean_distance --noise 0 --dataset mnist &
# python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg average --noise 0 --dataset mnist &
# python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg median --noise 0 --dataset mnist &
# python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg trimmed_mean --noise 0 --dataset mnist &
# python main_nn.py --model smallcnn --epochs 200 --gpu 0 --iid 1 --fix_total --frac 0.1 --num_attackers 20 --attacker_ep 5 --num_users 100 --attack_label 1 --agg krum --noise 0 --dataset mnist &
# wait

# # backdoor
# python main_nn.py --model smallcnn --epochs 200 --gpu 1 --iid 1 --local_bs 64 --num_attackers 20 --num_users 80 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg euclidean_distance --dataset mnist --donth_attack &
# python main_nn.py --model smallcnn --epochs 200 --gpu 1 --iid 1 --local_bs 64 --num_attackers 20 --num_users 80 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg average --dataset mnist --donth_attack &
# python main_nn.py --model smallcnn --epochs 200 --gpu 1 --iid 1 --local_bs 64 --num_attackers 20 --num_users 80 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg median --dataset mnist --donth_attack &
# python main_nn.py --model smallcnn --epochs 200 --gpu 1 --iid 1 --local_bs 64 --num_attackers 20 --num_users 80 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg trimmed_mean --dataset mnist --donth_attack &
# python main_nn.py --model smallcnn --epochs 200 --gpu 1 --iid 1 --local_bs 64 --num_attackers 20 --num_users 80 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg krum --dataset mnist --donth_attack &
# wait

!/bin/bash

python main_nn.py  --epochs 500 --gpu 0 --iid 1 --fix_total --frac 0.2 --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 --agg median --noise 0 --dataset cifar-100 &
python main_nn.py  --epochs 500 --gpu 0 --iid 1 --fix_total --frac 0.2 --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 --agg krum --noise 0 --dataset cifar-100 &
wait