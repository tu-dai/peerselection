import numpy as np


def precision_k(outcome, num_items, k):
    outcome = np.array(outcome)
    return np.sum(outcome >= num_items - k) / k


def positive_borda(outcome, num_items, k):
    outcome = np.array(outcome)
    opt = np.sum(range(k))
    scores = outcome - (num_items - k)
    scores[scores < 0] = 0
    return np.sum(scores) / opt


def negative_borda(outcome, num_items, k):
    outcome = np.array(outcome)
    swap_outcome = np.arange(num_items)

    swap_outcome = num_items - 1 - swap_outcome[~np.isin(swap_outcome, outcome)]
    new_k = num_items - k

    opt = np.sum(range(new_k))
    scores = swap_outcome - (num_items - new_k)
    scores[scores < 0] = 0
    return np.sum(scores) / opt