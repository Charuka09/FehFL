import copy
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import math


def add_gaussian_noise(w, scale):
    w_attacked = copy.deepcopy(w)
    if type(w_attacked) == list:
        for k in range(len(w_attacked)):
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    else:
        for k in w_attacked.keys():
            noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
            w_attacked[k] += noise
    return w_attacked


def model_poisoning_attack(w_local, w_glob, boost_factor=10.0):
    """Model poisoning: amplify the local update relative to the global model by boost_factor."""
    w_poisoned = copy.deepcopy(w_local)
    for k in w_poisoned.keys():
        delta = w_local[k].float() - w_glob[k].float()
        w_poisoned[k] = (w_glob[k].float() + boost_factor * delta).to(w_local[k].dtype)
    return w_poisoned


def min_max_attack(w_locals, attacker_positions, w_glob_state, gamma=2.0):
    """
    Min-max attack (simplified Shejwalkar et al. 2021).
    Perturbs attacker updates to maximally deviate from the benign mean
    while bounding magnitude to blend in with honest updates.
    """
    if not attacker_positions:
        return w_locals

    benign_positions = [i for i in range(len(w_locals)) if i not in attacker_positions]
    if not benign_positions:
        return w_locals

    # Compute coordinate-wise mean of benign updates
    w_benign_mean = copy.deepcopy(w_locals[benign_positions[0]])
    for k in w_benign_mean.keys():
        stacked = torch.stack([w_locals[i][k].float() for i in benign_positions])
        w_benign_mean[k] = stacked.mean(dim=0)

    result = list(copy.deepcopy(w) for w in w_locals)
    for pos in attacker_positions:
        for k in result[pos].keys():
            benign_mean_k = w_benign_mean[k]
            attacker_k = w_locals[pos][k].float()
            global_k = w_glob_state[k].float()

            # Direction: move attacker update away from benign mean
            direction = attacker_k - benign_mean_k
            norm = direction.norm()
            if norm < 1e-8:
                direction = torch.randn_like(direction)
                norm = direction.norm()
            direction = direction / norm

            # Scale by the magnitude of the honest update
            update_magnitude = (benign_mean_k - global_k).norm()
            result[pos][k] = (benign_mean_k + gamma * update_magnitude * direction).to(w_locals[pos][k].dtype)

    return result


def change_weight(w_attack, w_honest, change_rate=0.5):
    w_result = copy.deepcopy(w_honest)
    device = w_attack[list(w_attack.keys())[0]].device
    for k in w_honest.keys():
        w_h = w_honest[k]
        w_a = w_attack[k]

        assert w_h.shape == w_a.shape

        honest_idx = torch.FloatTensor((np.random.random(w_h.shape) > change_rate).astype(np.float)).to(device)
        attack_idx = torch.ones_like(w_h).to(device) - honest_idx

        weight = honest_idx * w_h + attack_idx * w_a
        w_result[k] = weight

    return w_result



