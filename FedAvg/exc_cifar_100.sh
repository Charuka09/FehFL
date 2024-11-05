!/bin/bash

python main_nn.py  --epochs 500 --gpu 2 --iid 1 --fix_total --frac 0.2 --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 --agg trimmed_mean --noise 0 --dataset cifar-100 &
wait

# python main_nn.py --epochs 1 --gpu 1 --iid 1 --local_bs 64 --num_attackers 10 --num_users 40 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 10 --agg euclidean_distance --dataset cifar-100 --donth_attack &
# python main_nn.py --epochs 500 --gpu 1 --iid 1 --local_bs 64 --num_attackers 10 --num_users 40 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 10 --agg average --dataset ifar-100  --donth_attack &
# python main_nn.py --epochs 500 --gpu 1 --iid 1 --local_bs 64 --num_attackers 10 --num_users 40 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 10 --agg median --dataset ifar-100  --donth_attack &
# python main_nn.py --epochs 500 --gpu 1 --iid 1 --local_bs 64 --num_attackers 10 --num_users 40 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg trimmed_mean --dataset ifar-100  --donth_attack &
# python main_nn.py --epochs 500 --gpu 1 --iid 1 --local_bs 64 --num_attackers 10 --num_users 40 --frac 0.1 --attacker_ep 10 --is_backdoor True --backdoor_label 1 --agg krum --dataset ifar-100  --donth_attack &