#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import math
import random
from scipy import stats
from functools import reduce
import time
import sklearn.metrics.pairwise as smp
from collections import OrderedDict
# import tensorflow as tf


eps = np.finfo(float).eps


def aggregate_weights(args, w_locals, net_glob, reweights, fg):
    # update global weights
    # choices are ['euclidean_distance', 'average', 'median', 
    #              'trimmed_mean', 'repeated', 'irls',
    #              'simple_irls', 'irls_median', 'irls_theilsen',
    #              'irls_gaussian', 'fg']
    agg_time = 0
    distList = []
    if args.agg == 'euclidean_distance':
        print("using euclidean distance average Estimator")
        w_glob, agg_time, distList = euclidean_distance_average(w_locals, net_glob)
    elif args.agg == 'median':
        print("using simple median Estimator")
        w_glob = simple_median(w_locals)
    elif args.agg == 'trimmed_mean':
        print("using trimmed mean Estimator")
        w_glob = trimmed_mean(w_locals, args.alpha)
    elif args.agg == 'krum':
        print("using simple Krum")
        w_glob, _  = krum(w_locals, 0)
    elif args.agg == 'repeated':
        print("using repeated median Estimator")
        w_glob, agg_time = Repeated_Median_Shard(w_locals)
    elif args.agg == 'irls':
        print("using IRLS Estimator")
        w_glob, reweight = IRLS_aggregation_split_restricted(w_locals, args.Lambda, args.thresh)
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'simple_irls':
        print("using simple IRLS Estimator")
        w_glob, reweight = simple_IRLS(w_locals, args.Lambda, args.thresh, args.alpha)
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'irls_median':
        print("using median IRLS Estimator")
        w_glob, reweight = IRLS_other_split_restricted(w_locals, args.Lambda, args.thresh, mode='median')
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'irls_theilsen':
        print("using TheilSen IRLS Estimator")
        w_glob, reweight = IRLS_other_split_restricted(w_locals, args.Lambda, args.thresh, mode='theilsen')
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'irls_gaussian':
        print("using Gaussian IRLS Estimator")
        w_glob, reweight = IRLS_other_split_restricted(w_locals, args.Lambda, args.thresh, mode='gaussian')
        print(reweight)
        reweights.append(reweight)
    elif args.agg == 'fg':
        # Update model
        # Add all the gradients to the model gradient
        net_glob.train()
        # train and update
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer.zero_grad()
        agg_grads = fg.aggregate_gradients(w_locals)
        for i, (name, params) in enumerate(net_glob.named_parameters()):
            if params.requires_grad:
                params.grad = agg_grads[i].cuda()
        optimizer.step()
    elif args.agg == 'average':
        print("using average")
        w_glob, agg_time = average_weights(w_locals)
    else:
        exit('Error: unrecognized aggregation method')
    return w_glob, agg_time, distList


def average_weights(w):
    cur_time = time.time()
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    print('model aggregation "average" took {}s'.format(time.time() - cur_time))
    agg_time = time.time() - cur_time
    return w_avg, agg_time


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]) / 2.0
    return output


def weighted_average(w_list, weights):
    w_avg = copy.deepcopy(w_list[0])
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(0, len(w_list)):
            w_avg[k] += w_list[i][k] * weights[i]
        # w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg, weights

def convert_tensor_to_np_arr(tensor):
    state_dict_cpu = {key: value.cpu() for key, value in tensor.items()}
    model_vector = torch.cat([value.view(-1) for value in state_dict_cpu.values()])
    model_vector = model_vector.detach().numpy()

    return model_vector

def special(w, global_model):
    dist_list = []
    w_local_total = []
    w_avg = copy.deepcopy(w[0])
    w_avg_copy = copy.deepcopy(w[0])
    total_num = 0 
    cur_time = time.time()
    device = w[0][list(w[0].keys())[0]].device
    total_dist = 0

    for k in w_avg.keys():
        shape = w_avg[k].shape
        print(k, shape)
        
        num_shape = reduce(lambda x, y: x * y, shape)
        total_num += num_shape

    y_list_g = global_model.to(device)
    w_global = y_list_g.state_dict()
    result_dict = OrderedDict()

    w_glob_data_arr = convert_tensor_to_np_arr(w_global)
    print(w_glob_data_arr, len(w_glob_data_arr))

    for i in range(len(w)):
        w_local_data_arr = convert_tensor_to_np_arr(w[i])
        dist = np.linalg.norm(w_glob_data_arr - w_local_data_arr)
        # print('dist..........', dist)
        dist_list.append(dist)
        for key in w[i]:
            w[i][key] *= 1/dist
        total_dist += (1/dist)
        for key in w_avg:
            result_dict[key] = w_avg[key] + w[i][key]
        # print(w_avg)
    # print(w_avg)
    for key in w_avg:
        result_dict[key] = w_avg[key] - w_avg_copy[key]
    values_list = list(w_avg.values())

    w_avg = torch.div(w_avg, total_dist)
    agg_time = time.time() - cur_time
    return w_avg, agg_time, dist_list


def euclidean_distance_average(w, global_model):
  dist_list = []
  w_local_total = []
  w, invalid_model_idx = get_valid_models(w)
  w_med = copy.deepcopy(w[0])
  w_avg = copy.deepcopy(w[0])
  w_avg_copy = copy.deepcopy(w[0])
  w_global = global_model.state_dict()
  cur_time = time.time()

  device = w[0][list(w[0].keys())[0]].device
  g_device = w_global[list(w[0].keys())[0]].device
  total_dist = 0

  for k in w_avg.keys():
    # print(k)
    shape = w_avg[k].shape
    # print(shape)
    if len(shape) == 0:
        continue
    total_num = reduce(lambda x, y: x * y, shape)
    y_list = torch.FloatTensor(len(w), total_num).to(device)
    y_list_g = torch.FloatTensor(len(w_global), total_num).to(g_device)
    y_list_g = torch.reshape(w_global[k], (-1,))
    y_g_t = torch.t(y_list_g)
    sub_list = []
    for i in range(len(w)):
        y_list[i] = torch.reshape(w[i][k], (-1,))
        # print(max(y_list[i]), min(y_list[i]))
        dist = np.linalg.norm(y_g_t.cpu().numpy() - y_list[i].cpu().numpy())
        if dist > 20 or math.isinf(dist) or math.isnan(dist):
            print(dist)
            dist = 20
        sub_list.append(dist)
        y = (1/dist)*y_list[i]
        total_dist += (1/dist)
        y = y.reshape(shape)
        w_avg[k] += y
    w_avg[k] = torch.div(w_avg[k], total_dist)
    dist_list += sub_list 
    total_dist = 0
  print('Fed Eucl Agg took {}s'.format(time.time() - cur_time))
  agg_time = time.time() - cur_time
  return w_avg, agg_time, dist_list

def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def gaussian_reweight_algorithm_restricted(y, sig, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)

    reweight = gaussian_zero_mean(e, sig=sig)
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def theilsen_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = theilsen(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + line_y * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def median_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    y_median = median_opt(y).unsqueeze(1).repeat(1, num_models)
    residual = y - y_median
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh).type(torch.cuda.FloatTensor) + y_median * (reweight < thresh).type(
        torch.cuda.FloatTensor)
    return reweight_regulized, restricted_y


def simple_reweight(y, LAMBDA, thresh, alpha):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    K = torch.FloatTensor([LAMBDA * np.sqrt(2. / num_models)]).to(y.device)

    beta = torch.cat((intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
                      slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2)), dim=-1)
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    # e = 1 / (residual.abs() + eps)
    # e_max = e.max(dim=-1)[0].unsqueeze(1).repeat(1, num_models)
    # reweight = e / e_max
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = 1 / e * torch.max(-K, torch.min(K, e))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = reweight * reshaped_std

    # sorted idx (remove alpha)
    sort_ids = e.abs().sort()[1].sort()[1]
    # remove_ids = sort_ids >= int((1 - alpha) * num_models)
    remove_ids = [i for i in sort_ids if i.item() >= int((1 - alpha) * num_models)]
    remove_ids = remove_ids * (reweight < thresh)
    keep_ids = (1 - remove_ids).type(torch.cuda.FloatTensor)
    remove_ids = remove_ids.type(torch.cuda.FloatTensor)
    restricted_y = y * keep_ids + line_y * remove_ids
    reweight_regulized = reweight_regulized * keep_ids
    return reweight_regulized, restricted_y


def is_valid_model(w):
    if isinstance(w, list):
        w_keys = list(range(len(w)))
    else:
        w_keys = w.keys()
    for k in w_keys:
        params = w[k]
        if torch.isnan(params).any():
            return False
        if torch.isinf(params).any():
            return False
    return True


def get_valid_models(w_locals):
    w, invalid_model_idx = [], []
    for i in range(len(w_locals)):
        if is_valid_model(w_locals[i]):
            w.append(w_locals[i])
        else:
            invalid_model_idx.append(i)
    return w, invalid_model_idx


def IRLS_aggregation_split_restricted(w_locals, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000
    cur_time = time.time()
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm_restricted(transposed_y_list, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm_restricted(y, LAMBDA, thresh)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    reweight = (reweight / reweight.max()).to(torch.device("cpu"))
    weights = torch.zeros(len(w_locals))
    i = 0
    for j in range(len(w_locals)):
        if j not in invalid_model_idx:
            weights[j] = reweight[i]
            i += 1

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, weights


def IRLS_other_split_restricted(w_locals, LAMBDA=2, thresh=0.1, mode='median'):
    if mode == 'median':
        reweight_algorithm = median_reweight_algorithm_restricted
    elif mode == 'theilsen':
        reweight_algorithm = theilsen_reweight_algorithm_restricted
    elif mode == 'gaussian':
        reweight_algorithm = gaussian_reweight_algorithm_restricted     # in gaussian reweight algorithm, lambda is sigma

    SHARD_SIZE = 2000
    cur_time = time.time()
    w, invalid_model_idx = get_valid_models(w_locals)
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = reweight_algorithm(transposed_y_list, LAMBDA, thresh)
            print(reweight.sum(dim=0))
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = reweight_algorithm(y, LAMBDA, thresh)
                print(reweight.sum(dim=0))
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    reweight = (reweight / reweight.max()).to(torch.device("cpu"))
    weights = torch.zeros(len(w_locals))
    i = 0
    for j in range(len(w_locals)):
        if j not in invalid_model_idx:
            weights[j] = reweight[i]
            i += 1

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, weights


def Repeated_Median_Shard(w):
    SHARD_SIZE = 100000
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        if total_num < SHARD_SIZE:
            slopes, intercepts = repeated_median(y)
            y = intercepts + slopes * (len(w) - 1) / 2.0
        else:
            y_result = torch.FloatTensor(total_num).to(device)
            assert total_num == y.shape[0]
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y_shard = y[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                slopes_shard, intercepts_shard = repeated_median(y_shard)
                y_shard = intercepts_shard + slopes_shard * (len(w) - 1) / 2.0
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE] = y_shard
            y = y_result
        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    agg_time = time.time() - cur_time
    return w_med, agg_time


def simple_IRLS(w, LAMBDA=2, thresh=0.03, alpha=1 / 11.0):
    SHARD_SIZE = 50000
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    # w_selected = [w[i] for i in random_select(len(w))]
    device = w[0][list(w[0].keys())[0]].device
    reweight_sum = torch.zeros(len(w)).to(device)

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        transposed_y_list = torch.t(y_list)
        y_result = torch.zeros_like(transposed_y_list)
        assert total_num == transposed_y_list.shape[0]

        if total_num < SHARD_SIZE:
            reweight, restricted_y = simple_reweight(transposed_y_list, LAMBDA, thresh, alpha)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = int(math.ceil(total_num / SHARD_SIZE))
            for i in range(num_shards):
                y = transposed_y_list[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...]
                reweight, restricted_y = simple_reweight(y, LAMBDA, thresh, alpha)
                reweight_sum += reweight.sum(dim=0)
                y_result[i * SHARD_SIZE: (i + 1) * SHARD_SIZE, ...] = restricted_y

        # put restricted y back to w
        y_result = torch.t(y_result)
        for i in range(len(w)):
            w[i][k] = y_result[i].reshape(w[i][k].shape).to(device)
        # print(reweight_sum  )
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    print('model aggregation took {}s'.format(time.time() - cur_time))
    return w_med, (reweight / reweight.max()).to(torch.device("cpu"))


def random_select(size, thresh=0.5):
    assert thresh < 1.0
    a = []
    while len(a) < 3:
        for i in range(size):
            if random.uniform(0, 1) > thresh:
                a.append(i)
    return a


def theilsen(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yy = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyj = yy
    yyi = yyj.transpose(-1, -2)
    xx = torch.cuda.FloatTensor(range(num_models))
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.cuda.FloatTensor([float('Inf')] * num_models)
    inf_lower = torch.tril(diag.repeat(num_models, 1), diagonal=0).repeat(total_num, 1, 1)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + inf_lower
    slopes, _ = torch.flatten(slopes, 1, 2).sort()
    raw_slopes = slopes[:, :int(num_models * (num_models - 1) / 2)]
    slopes = median_opt(raw_slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.cuda.FloatTensor(xx_median)
    intercepts = yy_median - slopes * xx_median
    return slopes, intercepts


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float('Inf')] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)

    # get intercepts (intercept of median)
    yy_median = median_opt(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


# Repeated Median estimator
def Repeated_Median(w):
    cur_time = time.time()
    w_med = copy.deepcopy(w[0])
    device = w[0][list(w[0].keys())[0]].device

    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)

        slopes, intercepts = repeated_median(y)
        y = intercepts + slopes * (len(w) - 1) / 2.0

        y = y.reshape(shape)
        w_med[k] = y

    print('repeated median aggregation took {}s'.format(time.time() - cur_time))
    return w_med


# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
    n_clients = grads.shape[0]
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv


class FoolsGold(object):
    def __init__(self, args):
        self.memory = None
        self.wv_history = []
        self.args = args

    def aggregate_gradients(self, client_grads):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        if self.memory is None:
            self.memory = np.zeros((num_clients, grad_len))

        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))


        if self.args.use_memory:
            self.memory += grads
            wv = foolsgold(self.memory)  # Use FG
        else:
            wv = foolsgold(grads)  # Use FG
        print(wv)
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(
                len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads


# simple median estimator
def simple_median(w):
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    cur_time = time.time()
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        median_result = median_opt(y)
        assert total_num == len(median_result)

        weight = torch.reshape(median_result, shape)
        w_med[k] = weight
    print('model aggregation "median" took {}s'.format(time.time() - cur_time))
    return w_med


def trimmed_mean(w, trim_ratio):
    assert trim_ratio < 0.5, 'trim ratio is {}, but it should be less than 0.5'.format(trim_ratio)
    trim_num = int(trim_ratio * len(w))
    device = w[0][list(w[0].keys())[0]].device
    w_med = copy.deepcopy(w[0])
    cur_time = time.time()
    for k in w_med.keys():
        shape = w_med[k].shape
        if len(shape) == 0:
            continue
        total_num = reduce(lambda x, y: x * y, shape)
        y_list = torch.FloatTensor(len(w), total_num).to(device)
        for i in range(len(w)):
            y_list[i] = torch.reshape(w[i][k], (-1,))
        y = torch.t(y_list)
        y_sorted = y.sort()[0]
        result = y_sorted[:, trim_num:-trim_num]
        result = result.mean(dim=-1)
        assert total_num == len(result)

        weight = torch.reshape(result, shape)
        w_med[k] = weight
    print('model aggregation "trimmed mean" took {}s'.format(time.time() - cur_time))
    return w_med

def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)

def multi_vectorization(w_locals, device):
    vectors = copy.deepcopy(w_locals)
    
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(device)
        vectors[i] = torch.cat(list(v.values()))
    return vectors

def pairwise_distance(w_locals, device):
    
    vectors = multi_vectorization(w_locals, device)
    distance = torch.zeros([len(vectors), len(vectors)]).to(device)
    
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
                
    return distance

def krum(w, c):    
    n = len(w) - c
    device = w[0][list(w[0].keys())[0]].device

    distance = pairwise_distance(w, device)
    sorted_idx = distance.sum(dim=0).argsort()[: n]

    chosen_idx = int(sorted_idx[0])
        
    return copy.deepcopy(w[chosen_idx]), chosen_idx

def gaussian_zero_mean(x, sig=1):
    return torch.exp(- x * x / (2 * sig * sig))


if __name__ == "__main__":
    # from matplotlib import pyplot as mp
    #
    # x_values = np.linspace(-3, 3, 120)
    # for mu, sig in [(0, 1)]:
    #     mp.plot(x_values, gaussian(x_values, mu, sig))
    #
    # mp.show()

    torch.manual_seed(0)
    y = torch.ones(1, 10).cuda()
    e = gaussian_reweight_algorithm_restricted(y, 2, thresh=0.1)
    print(y)
    print(e)
