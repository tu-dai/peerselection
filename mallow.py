import numpy as np
import random

from tqdm import tqdm


def computeInsertionProbas(i, phi):
    probas = (i + 1) * [0]
    for j in range(i + 1):
        probas[j] = pow(phi, (i + 1) - (j + 1))
    return probas


def weighted_choice(choices):
    total = 0
    for w in choices:
        total = total + w
    r = np.random.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(choices):
        if upto + w >= r:
            return i
        upto = upto + w
    assert False, "Shouldn't get here"


def mallowsVote(m, insertion_probabilites_list):
    vote = [0]
    for i in range(1, m):
        index = weighted_choice(insertion_probabilites_list[i - 1])
        vote.insert(index, i)
    return vote

def generate_mallows(n,m,phi, show_progress=False):
    insertion_probabilites_list = []
    for i in range(1, m):
        insertion_probabilites_list.append(computeInsertionProbas(i, phi))
    V = []
    if show_progress:
        for i in tqdm(range(n), desc=" inner loop", position=1, leave=False):
            vote = mallowsVote(m, insertion_probabilites_list)
            V += [vote]
    else:
        for i in range(n):
            vote = mallowsVote(m, insertion_probabilites_list)
            V += [vote]
    return np.array(V)



#########################

def mallows_sample(n_items, phi, base_rank):
    if phi == 1:
        # If phi is 1, return the base rank because all rankings are equally probable.
        return base_rank.copy()

    ranking = base_rank.copy()
    for i in range(n_items - 1):
        # Calculate the probabilities
        weights = [phi ** (j - i) for j in range(i, n_items)]
        weights = np.array(weights) / np.sum(weights)

        # Select an index according to the probabilities
        selected_index = np.random.choice(np.arange(i, n_items), p=weights)

        # Swap the elements to create the ranking
        ranking[i], ranking[selected_index] = ranking[selected_index], ranking[i]

    return ranking

def generate_mallows_rankings(N, n_items, phi):
    base_rank = list(range(n_items))
    rankings = np.array([mallows_sample(n_items, phi, base_rank) for _ in range(N)])
    return rankings
