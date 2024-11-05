!/bin/bash

python main_nn.py  --epochs 500 --gpu 1 --iid 1 --fix_total --frac 0.2 --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 --agg euclidean_distance --noise 0 --dataset cifar-100 &
python main_nn.py  --epochs 500 --gpu 1 --iid 1 --fix_total --frac 0.2 --num_attackers 20 --attacker_ep 5 --num_users 50 --attack_label 10 --agg average --noise 0 --dataset cifar-100 &
wait