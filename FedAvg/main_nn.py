#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from collections import deque
from tqdm import tqdm
import torch
from datasets import build_datasets
from tensorboardX import SummaryWriter
from options import args_parser
from Update import LocalUpdate
from FedNets import build_model
from averaging import aggregate_weights, get_valid_models, FoolsGold, IRLS_aggregation_split_restricted
from attack import add_gaussian_noise, change_weight
import json
import sys


def test(net_g, dataset, args, dict_users):
    # testing
    list_acc = []
    list_loss = []
    net_test = copy.deepcopy(net_g)
    net_test.eval()
    if args.dataset == "mnist":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif args.dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == "loan":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8')
    elif args.dataset == "cifar-100":
        classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
           'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
           'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
           'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
           'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
           'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
           'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
           'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
           'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
           'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
           'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
           'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
           'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
           'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    with torch.no_grad():
        added_acc=0
        added_loss=0
        added_data_num=0
        for c in range(len(classes)):
            if c < len(classes) and len(dict_users[c])>0:

                net_local = LocalUpdate(args=args, dataset=dataset, idxs=dict_users[c], tb=None, test_flag=True)
                acc, loss = net_local.test(net=net_test)
                # print("test accuracy for label {} is {:.2f}%".format(classes[c], acc * 100.))
                list_acc.append(acc)
                list_loss.append(loss)
                added_acc+= acc*len(dict_users[c])
                added_loss+= loss*len(dict_users[c])
                added_data_num+=len(dict_users[c])

        print("average acc: {:.2f}%,\ttest loss: {:.5f}".format(100. * added_acc/ float(added_data_num),
                                                                added_loss/ float(added_data_num)))

    return list_acc, added_acc/ float(added_data_num)


def backdoor_test(net_g, dataset, args, idxs):
    # backdoor testing
    net_test = copy.deepcopy(net_g)
    net_test.eval()
    with torch.no_grad():
        net_local = LocalUpdate(args=args, dataset=dataset, idxs=idxs, tb=None,
                                backdoor_label=args.backdoor_label, test_flag=True)
        acc, loss = net_local.backdoor_test(net=net_test)
        print("backdoor acc: {:.2f}%,\ttest loss: {:.5f}".format(100. * acc, loss))
    return acc


if __name__ == '__main__':
    # parse args
    args = args_parser()
    np.random.seed(args.seed)
    learning_rate = args.lr
    # set attack mode
    print('perform poison attack with {} attackers'.format(args.num_attackers))

    # define paths
    path_project = os.path.abspath('..')
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)

    summary = SummaryWriter('local')

    dataset_train, dataset_test, dict_users, test_users, attackers = build_datasets(args)

    if not args.fix_total:
        args.num_users += args.num_attackers

    # check poison attack
    if args.dataset == 'mnist' and args.num_attackers > 0:
        assert args.attack_label == 1 or (args.donth_attack and args.attack_label < 0)
    elif args.dataset == 'cifar' and args.num_attackers > 0:
        assert args.attack_label == 3 or (args.donth_attack and args.attack_label < 0)
    elif args.dataset == 'loan' and args.num_attackers > 0:
        assert args.attack_label == 0
    elif args.dataset == 'cifar-100' and args.num_attackers > 0:
        assert args.attack_label == 10 or (args.donth_attack and args.attack_label < 0)

    # build model
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
    net_glob = build_model(args)

    # init FoolsGold
    if args.agg == 'fg':
        fg = FoolsGold(args)
    else:
        fg = None

    # copy weights
    w_glob = net_glob.state_dict()
    net_glob.train()

    ######################################################
    # Training                                           #
    ######################################################
    loss_train = []
    accs = []
    att_acc_list = []
    avg_acc = []
    reweights = []
    backdoor_accs = []
    total_agg_time = 0
    distances_actual = []
    distances_malicious = []
    if 'irls' in args.agg:
        model_save_path = '../weights/{}_{}_irls_{}_{}_{}'.format(args.dataset, args.model, args.Lambda, args.thresh, args.iid)
    else:
        model_save_path = '../weights/{}_{}'.format(args.dataset, args.model)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    last_bd_acc=0
    lr_change_ratio=10.0
    idx_queue = deque()
    label_idxs = np.arange(60000)
    entrpopies = []
    labels = None
    #update for now ---
    # net_glob.train()
    if args.is_dynamic:
        labels = dataset_train.train_labels.numpy()
        print('len(labels)', len(labels))
        for i in range(50000,60000):
            idx_queue.append(i)
        print('idx_queue[0]',idx_queue[0], len(idx_queue))
    for iter in tqdm(range(args.epochs)):
        iter_entropies = []
        print('Epoch:', iter, "/", args.epochs)
        net_glob.train()
        w_locals, w_locals_noise, loss_locals = [], [], []
        m = max(int(args.frac * args.num_users), 1)
        # print('taking {} users'.format(m))
        if args.frac == 1:
            idxs_users = range(args.num_users)
        else:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            w_noise = None
            if args.is_dynamic:
                # Update each user with new data from the data queue
                idx_update = [idx_queue.popleft() for _ in range(10)]
                idx_update = np.array(idx_update)
                if isinstance(dict_users[idx], set):
                    dict_users[idx] = np.array(list(dict_users[idx]))
                dict_users[idx] = np.concatenate((dict_users[idx], idx_update), axis=0)
                user_labels = [labels[np.where(label_idxs == item)][0] for item in dict_users[idx]]
                bin_count = np.bincount(user_labels)
                mask = bin_count == 0
                random_values = np.random.randint(1, 2, size=(bin_count.shape))
                bin_count[mask] = random_values[mask]
                num_labels = np.sum(np.arange(len(bin_count)) * bin_count)
                probabilities = bin_count / num_labels
                entropy = -np.sum(probabilities * np.log2(probabilities))
                iter_entropies.append({idx: entropy})
                # print(idx)
            # Train the new without any attackers
            if (idx >= args.num_users - args.num_attackers and not args.fix_total) or \
                    (args.fix_total and idx in attackers):
                print('id and attackers', idx, attackers)
            # if (idx >= args.num_users - args.num_attackers and not args.fix_total):
                local_ep = args.local_ep
                if args.attacker_ep != args.local_ep:
                    if args.dataset == 'loan':
                        lr_change_ratio = 1.0/5
                    args.lr = args.lr * lr_change_ratio
                args.local_ep = args.attacker_ep

                if args.is_backdoor:
                    if args.backdoor_single_shot_scale_epoch ==-1 or iter == args.backdoor_single_shot_scale_epoch:
                        print('backdoor attacker', idx)
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary,
                                        backdoor_label=args.backdoor_label)
                    else: # under one-single-shot mode; don't perform backdoor attack because it's not the scale epoch:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
                else:
                    if args.fix_total and args.attack_label < 0 and args.donth_attack:
                        continue
                    else:
                        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary,
                                        attack_label=args.attack_label)

                temp_net = copy.deepcopy(net_glob)
                if args.agg == 'fg':
                    w, loss, _poisoned_net = local.update_gradients(net=temp_net)
                elif args.agg == 'lsfe':
                    w, w_1, w_2, loss, updated_net = local.update_weights_with_lsfe(net=temp_net)
                elif args.noise:
                    w, w_noise, loss, _updated_net = local.update_weights_with_noise(net=temp_net)
                    print([w, w_noise],  file=open('./results_noise/results.txt', 'w'))
                else:
                    w, loss, _poisoned_net = local.update_weights(net=temp_net)

                # change a portion of the model gradients to honest
                if 0 < args.change_rate < 1.:
                    w_honest, reweight = IRLS_aggregation_split_restricted(w_locals, args.Lambda, args.thresh)
                    w = change_weight(w, w_honest, change_rate=args.change_rate)
                args.local_ep = local_ep
                if args.attacker_ep != args.local_ep:
                    args.lr = args.lr / lr_change_ratio
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
                temp_net = copy.deepcopy(net_glob)
                if args.agg == 'fg':
                    w, loss, _updated_net = local.update_gradients(net=temp_net)
                elif args.agg == 'lsfe':
                    w, w_1, w_2, loss, updated_net = local.update_weights_with_lsfe(net=temp_net)
                elif args.noise:
                    w, w_noise, loss, _updated_net = local.update_weights_with_noise(net=temp_net)
                    w_locals_noise, invalid_model_idx_noise = get_valid_models(w_locals)
                    # print("invalid_model_idx_noise--------", invalid_model_idx_noise)
                    print([w, w_noise],  file=open('./results_noise/results.txt', 'w'))
                else:
                    w, loss, _updated_net = local.update_weights(net=temp_net)
                                
            w_locals_noise.append(copy.deepcopy(w_noise))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # remove model with inf values
        w_locals, invalid_model_idx= get_valid_models(w_locals)
        if invalid_model_idx:
            print("invalid_model_idx--------", invalid_model_idx, len(w_locals))
        entrpopies.append(iter_entropies)
        if len(w_locals) == 0:
            break
            continue
        w_glob, agg_time, dist_list_actual, dist_list_malicious = aggregate_weights(args, w_locals, w_locals_noise, net_glob, reweights, fg)
        total_agg_time += agg_time
        print('Aggregation time - ', agg_time)
        distances_actual += dist_list_actual
        distances_malicious += dist_list_malicious
        net_glob.load_state_dict(w_glob)
        # copy weight to net_glob
        if not args.agg == 'fg':
            net_glob.load_state_dict(w_glob)

        # test data
        list_acc, avg_acc_list = test(net_glob, dataset_test, args, test_users)
        accs.append(list_acc)
        avg_acc.append(avg_acc_list)

        if args.attack_label == -1:
            args.attack_label = 1
        net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=test_users[args.attack_label], tb=summary, attack_label=args.attack_label,
                                    test_flag=True)
        att_acc, att_loss = net_local.test(net=net_glob)
        if args.num_attackers > 0:
            print("attack success rate for attacker is {:.2f}%".format(att_acc * 100.))
        att_acc_list.append(att_acc)
        # poisoned test data
        if args.is_backdoor:
            _backdoor_acc = backdoor_test(net_glob, dataset_test, args, np.asarray(list(range(len(dataset_test)))))
            backdoor_accs.append([_backdoor_acc])
            last_bd_acc = _backdoor_acc
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        if args.epochs % 10 == 0:
            print('\nTrain loss:', loss_avg)
        loss_train.append(loss_avg)

    print('backdoor_acc.....', backdoor_accs) 
    print(entrpopies,  file=open('./entropies.txt', 'w'))
    save_folder='./save/'
    results='./results/'

    print('distances_actual: ', distances_actual)
    print('distances_malicious: ', distances_malicious)

    # writing distance outputs to a json file
    # print(distances,  file=open('./distances.json', 'w'))

    ######################################################
    # Testing                                            #
    ######################################################
    list_acc, list_loss = [], []
    net_glob.eval()
    if args.dataset == "mnist":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    elif args.dataset == "cifar":
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif args.dataset == "loan":
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8')
    elif args.dataset == "cifar-100":
        classes = ('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
           'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
           'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
           'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
           'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
           'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
           'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
           'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
           'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
           'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
           'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
           'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
           'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
           'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
    added_acc = 0
    added_loss = 0
    added_data_num = 0
    with torch.no_grad():
        for c in range(len(classes)):
            if c < len(classes) and len(test_users[c])>0:
                net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=test_users[c], tb=summary, test_flag=True)
                acc, loss = net_local.test(net=net_glob)
                print("test accuracy for label {} is {:.2f}%".format(classes[c], acc * 100.))
                list_acc.append(acc)
                list_loss.append(loss)
                added_acc += acc * len(test_users[c])
                added_loss += loss * len(test_users[c])
                added_data_num += len(test_users[c])
        print("average acc: {:.2f}%,\ttest loss: {:.5f}".format(100. * added_acc / float(added_data_num),
                                                                added_loss / float(added_data_num)))
        if args.is_backdoor:
            acc = backdoor_test(net_glob, dataset_test, args, np.asarray(list(range(len(dataset_test)))))
            print("backdoor success rate for attacker is {:.2f}%".format(acc * 100.))
        else:
            if args.attack_label == -1:
                args.attack_label = 1
            net_local = LocalUpdate(args=args, dataset=dataset_test, idxs=test_users[args.attack_label], tb=summary, attack_label=args.attack_label,
                                    test_flag=True)
            acc, loss = net_local.test(net=net_glob)
            print("attack success rate for attacker is {:.2f}%".format(acc * 100.))
    
    ######################################################
    # Plot                                               #
    ######################################################
   
    # Plot overall accuracy by each epoch
    plt.figure()
    plt.title('Accuracy against epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.plot(range(1, len(avg_acc)+1), avg_acc, color='green',
         marker='x', linestyle='dashed', linewidth=1.2, markersize=4, markevery=4)
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.savefig(
        save_folder+'avg_acc_{}_{}_{}_{}_users{}_attackers_{}_attackep_{}_thresh_{}_iid{}_noise{}.png'.format(args.agg,args.dataset,
                                                                                           args.model, args.epochs,
                                                                                           args.num_users - args.num_attackers,
                                                                                           args.num_attackers,
                                                                                           args.attacker_ep,
                                                                                           args.thresh,args.iid, args.noise))
    print('avg_acc--', avg_acc)
    print('Average agg time: ', total_agg_time/args.epochs)
    print('Total agg time:', total_agg_time)
    # plot acc by class
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    accs_np = np.zeros((len(accs), len(accs[0])))
    for i, w in enumerate(accs):
        accs_np[i] = np.array(w)
    accs_np = accs_np.transpose()
    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 0.9, len(accs_np))]
    plt.title('Accuracy by class against epochs')
    plt.ylabel('accuracy'.format(i))
    plt.xlabel('epoch')
    for i, y in enumerate(accs_np):
        plt.plot(range(len(y)), y)
    for i, j in enumerate(ax1.lines):
        j.set_color(colors[i])
    plt.legend([str(i) for i in range(len(accs_np))], loc='lower right')
    plt.savefig(
        save_folder+'acc_{}_{}_{}_{}_users{}_attackers_{}_attackep_{}_thresh_{}_iid{}_noise{}.png'.format(args.agg,args.dataset,
                                                                                           args.model, args.epochs,
                                                                                           args.num_users - args.num_attackers,
                                                                                           args.num_attackers,
                                                                                           args.attacker_ep,
                                                                                           args.thresh,args.iid, args.noise))
    # plot loss curve
    plt.figure()
    plt.plot(range(1, len(loss_train)+1), loss_train, color='red',
         marker='x', linestyle='dashed', linewidth=1.2, markersize=4, markevery=4)
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.title('Training Loss against epochs')
    plt.ylabel('train_loss')
    plt.xlabel('epoch')
    plt.savefig(save_folder+'fed_{}_{}_{}_{}_C{}_iid{}_noise{}.png'.format(args.agg,args.dataset, args.model, args.epochs, args.frac, args.iid, args.noise))

    # Saving results into files
    if args.agg == "euclidean_distance":
        results = './results/e_d/'
        # print(loss_train,  file=open('./results/e_d/loss.txt', 'w'))
        # print(att_acc_list,  file=open('./results/e_d/asr.txt', 'w'))
        # print(avg_acc,  file=open('./results/e_d/accuracy.txt', 'w'))
    elif args.agg == "average":
        results = './results/fedAvg/'
        print(loss_train,  file=open('./results/fedAvg/loss.txt', 'w'))
        print(att_acc_list,  file=open('./results/fedAvg/asr.txt', 'w'))
        print(avg_acc,  file=open('./results/fedAvg/accuracy.txt', 'w'))
    elif args.agg == "median":
        results = './results/median/'
        print(loss_train,  file=open('./results/median/loss.txt', 'w'))
        print(att_acc_list,  file=open('./results/median/asr.txt', 'w'))
        print(avg_acc,  file=open('./results/median/accuracy.txt', 'w'))
    elif args.agg == "trimmed_mean":
        results = './results/mean/'
        print(loss_train,  file=open('./results/mean/loss.txt', 'w'))
        print(att_acc_list,  file=open('./results/mean/asr.txt', 'w'))
        print(avg_acc,  file=open('./results/mean/accuracy.txt', 'w'))
    elif args.agg == "krum":
        results = './results/krum/'
        print(loss_train,  file=open('./results/krum/loss.txt', 'w'))
        print(att_acc_list,  file=open('./results/krum/asr.txt', 'w'))
        print(avg_acc,  file=open('./results/krum/accuracy.txt', 'w'))
    # print('loss_train--', loss_train)
    try:
        os.makedirs(results, exist_ok=True)
        print(f"Directory {results} is ready.")
    except Exception as e:
        print(f"Error creating directory: {e}")

    # Construct file paths dynamically
    try:
        accuracy_filename = os.path.join(results, 'accuracy_{}_{}.txt'.format(
            args.dataset, args.num_attackers
        ))
        asr_filename = os.path.join(results, 'asr_{}_{}.txt'.format(
            args.dataset, args.num_attackers
        ))

        print(f"Accuracy filename: {accuracy_filename}")
        print(f"ASR filename: {asr_filename}")

        # Save accuracy to the dynamically named .txt file
        with open(accuracy_filename, 'w') as f:
            print(f"Writing accuracy to {accuracy_filename}")
            print(avg_acc, file=f)

        # Save ASR list to the dynamically named .txt file
        with open(asr_filename, 'w') as f:
            print(f"Writing ASR to {asr_filename}")
            print(att_acc_list, file=f)

        print("Files created and written successfully.")

    except Exception as e:
        print(f"Error writing to file: {e}")

    if args.is_backdoor:
        # plot backdoor acc
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        backdoor_accs_np = np.zeros((len(backdoor_accs), len(backdoor_accs[0])))
        for i, w in enumerate(backdoor_accs):
            backdoor_accs_np[i] = np.array(w)
        backdoor_accs_np = backdoor_accs_np.transpose()
        colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
        colors = [colormap(i) for i in np.linspace(0, 0.9, len(backdoor_accs_np))]
        plt.ylabel('backdoor success rate'.format(i))
        for i, y in enumerate(backdoor_accs_np):
            plt.plot(range(len(y)), y)
        for i, j in enumerate(ax1.lines):
            j.set_color(colors[i])
        plt.legend([str(i) for i in range(len(backdoor_accs_np))], loc='lower right')
        plt.savefig(
            save_folder+'backdoor_accs_{}_{}_{}_{}_users{}_attackers_{}_attackep_{}_thresh_{}_iid{}.png'.format(args.agg,args.dataset,
                                                                                                         args.model,
                                                                                                         args.epochs,
                                                                                                         args.num_users - args.num_attackers,
                                                                                                         args.num_attackers,                                                                                                   args.attacker_ep,
                                                                                                         args.thresh, args.iid))
