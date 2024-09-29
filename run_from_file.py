import pickle
import time

import h5py
import numpy as np
import random
from tqdm import tqdm

from create_edp import create_profile
from mallow import generate_mallows
from metrics import precision_k, positive_borda, negative_borda
from peerselection.peerselect.peerselect import impartial, profile_generator

import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_probs(outputs, num_profiles, num_items):
    flattened = np.concatenate(outputs)
    vals, counts = np.unique(flattened, return_counts=True)
    cur_probs = np.zeros(num_items)
    cur_probs[vals] = counts / num_profiles
    return cur_probs

def run_experiment(profile, clustering, assignment_1, assignment_20, scores, k):
    score_matrix = profile_generator.profile_classes_to_score_matrix(profile, scores)
    score_matrix_1 = profile_generator.restrict_score_matrix(score_matrix, assignment_1)
    score_matrix_20 = profile_generator.restrict_score_matrix(score_matrix, assignment_20)

    # # Run vanilla
    # vanilla_1 = impartial.vanilla(score_matrix_1, k)
    # vanilla_20 = impartial.vanilla(score_matrix_20, k)

    # Run edp
    edp_1 = impartial.exact_dollar_partition_explicit(score_matrix_1, k, clustering, normalize=True)
    edp_20 = impartial.exact_dollar_partition_explicit(score_matrix_20, k, clustering, normalize=True)

    # Run partition
    partition_1 = impartial.partition_explicit(score_matrix_1, k, clustering, normalize=False)
    partition_20 = impartial.partition_explicit(score_matrix_20, k, clustering, normalize=False)

    # # Run peer nomination
    # peer_1 = impartial.peer_nomination_lottery(score_matrix_1, k )
    # peer_20 = impartial.peer_nomination_lottery(score_matrix_20, k)

    # return vanilla_1, vanilla_20, edp_1, edp_20, partition_1, partition_20, peer_1, peer_20
    return edp_1, edp_20, partition_1, partition_20


def calc_metrics(outcome, num_items, k):
    prec = precision_k(outcome, num_items, k)
    pos_borda = positive_borda(outcome, num_items, k)
    neg_borda = negative_borda(outcome, num_items, k)
    return prec, pos_borda, neg_borda

def main():
    num_exps = 10000
    num_items = 200
    items = np.arange(0, num_items)
    scores = np.arange(0, num_items)
    k = 50

    res_vanilla = {}
    res_partition = {}
    res_edp = {}
    res_peer = {}

    phi = 0.2
    phi_str = str(phi).replace(".", "")

    print("loading profiles")
    with gzip.open(f'profiles_{phi_str}.gzip', 'rb') as f:
        profiles = pickle.load(f)

    print("starting to run")

    for num_clusters in tqdm([3, 10, 20, 50], desc='#clusters', leave=False, position=0):
        vanilla_1 = []
        vanilla_20 = []
        vanilla1_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        vanilla20_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        partition_1 = []
        partition_20 = []
        partition1_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        partition20_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        edp_1 = []
        edp_20 = []
        edp1_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        edp20_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        peer_1 = []
        peer_20 = []
        peer1_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
        peer20_metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}

        with gzip.open(f'partition_{num_clusters}.gzip', 'rb') as f:
            part_data = pickle.load(f)

        clusters = part_data['clusters']
        assignments_1 = part_data['assignments_1']
        assignments_20 = part_data['assignments_20']

        # Parallelizing the experiment runs
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_exps):
                profile = profiles[i]
                clustering = clusters[i]
                assignment_1 = assignments_1[i]
                assignment_20 = assignments_20[i]
                futures.append(executor.submit(run_experiment, profile, clustering, assignment_1, assignment_20, scores, k))

            for future in tqdm(as_completed(futures), total=num_exps, desc='Experiments', leave=False, position=1):
                # vanilla_1_result, vanilla_20_result, edp_1_result, edp_20_result, \
                # partition_1_result, partition_20_result, peer_1_result, peer_20_result = future.result()
                edp_1_result, edp_20_result, partition_1_result, partition_20_result = future.result()
                # vanilla_1.append(vanilla_1_result)
                # vanilla_20.append(vanilla_20_result)
                partition_1.append(partition_1_result)
                partition_20.append(partition_20_result)
                edp_1.append(edp_1_result)
                edp_20.append(edp_20_result)
                # peer_1.append(peer_1_result)
                # peer_20.append(peer_20_result)

                # for outcome, mdict in zip([vanilla_1_result, vanilla_20_result, edp_1_result, edp_20_result,
                #                            partition_1_result, partition_20_result, peer_1_result, peer_20_result],
                #                           [vanilla1_metrics, vanilla20_metrics, partition1_metrics, partition20_metrics,
                #                            edp1_metrics, edp20_metrics, peer1_metrics, peer20_metrics]):
                for outcome, mdict in zip([partition_1_result, partition_20_result, edp_1_result, edp_20_result],
                                          [partition1_metrics, partition20_metrics, edp1_metrics, edp20_metrics]):
                    prec, pb, nb = calc_metrics(outcome, num_items, k)
                    mdict['precision@k'].append(prec)
                    mdict['positive_borda'].append(pb)
                    mdict['negative_borda'].append(nb)


        # vanilla1_probs = get_probs(vanilla_1, num_exps, num_items)
        # vanilla20_probs = get_probs(vanilla_20, num_exps, num_items)
        partition1_probs = get_probs(partition_1, num_exps, num_items)
        partition20_probs = get_probs(partition_20, num_exps, num_items)
        edp1_probs = get_probs(edp_1, num_exps, num_items)
        edp20_probs = get_probs(edp_20, num_exps, num_items)
        # peer1_probs = get_probs(peer_1, num_exps, num_items)
        # peer20_probs = get_probs(peer_20, num_exps, num_items)

        # for method, mdict, mname in zip([edp_1, edp_20, partition_1, partition_20],
        #                                 [edp1_metrics, edp20_metrics, partition1_metrics, partition20_metrics],
        #                                 []):
        #     for outcome in method:
        #         prec, pb, nb = calc_metrics(outcome, num_items, k)
        #         mdict['precision@k'].append(prec)
        #         mdict['positive_borda'].append(pb)
        #         mdict['negative_borda'].append(nb)

        res_vanilla[num_clusters] = {}
        res_partition[num_clusters] = {}
        res_edp[num_clusters] = {}
        res_peer[num_clusters] = {}
        # res_vanilla[num_clusters]['gains'] = (vanilla20_probs - vanilla1_probs).tolist()
        res_partition[num_clusters]['gains'] = (partition20_probs - partition1_probs).tolist()
        res_partition[num_clusters]['probs'] = {'1': partition1_probs.tolist(),
                                                '20': partition20_probs.tolist()}
        res_edp[num_clusters]['gains'] = (edp20_probs - edp1_probs).tolist()
        res_edp[num_clusters]['probs'] = {'1': edp1_probs.tolist(),
                                                '20': edp20_probs.tolist()}
        # res_peer[num_clusters]['gains'] = (peer20_probs - peer1_probs).tolist()


        # cur_result = {}
        # for sample_size, size_metrics in zip(['1', '20'], [vanilla1_metrics, vanilla20_metrics]):
        #     cur_result[sample_size] = {}
        #     for metric_name in ['precision@k', 'positive_borda', 'negative_borda']:
        #         cur_result[sample_size][metric_name] = {'mean': np.mean(size_metrics[metric_name]),
        #                                                 'std': np.std(size_metrics[metric_name])}
        #
        # res_vanilla[num_clusters]['metrics'] = cur_result

        cur_result = {}
        for sample_size, size_metrics in zip(['1', '20'], [edp1_metrics, edp20_metrics]):
            cur_result[sample_size] = {}
            for metric_name in ['precision@k', 'positive_borda', 'negative_borda']:
                cur_result[sample_size][metric_name] = {'mean': np.mean(size_metrics[metric_name]),
                                                        'std': np.std(size_metrics[metric_name])}

        res_edp[num_clusters]['metrics'] = cur_result

        cur_result = {}
        for sample_size, size_metrics in zip(['1', '20'], [partition1_metrics, partition20_metrics]):
            cur_result[sample_size] = {}
            for metric_name in ['precision@k', 'positive_borda', 'negative_borda']:
                cur_result[sample_size][metric_name] = {'mean': np.mean(size_metrics[metric_name]),
                                                        'std': np.std(size_metrics[metric_name])}

        res_partition[num_clusters]['metrics'] = cur_result

        # cur_result = {}
        # for sample_size, size_metrics in zip(['1', '20'], [peer1_metrics, peer20_metrics]):
        #     cur_result[sample_size] = {}
        #     for metric_name in ['precision@k', 'positive_borda', 'negative_borda']:
        #         cur_result[sample_size][metric_name] = {'mean': np.mean(size_metrics[metric_name]),
        #                                                 'std': np.std(size_metrics[metric_name])}
        #
        # res_peer[num_clusters]['metrics'] = cur_result

        # res_edp[num_clusters] = (edp20_probs - edp1_probs).tolist()
        # res_partition[num_clusters] = (partition20_probs - partition1_probs).tolist()

        del clusters
        del assignments_1
        del assignments_20
        del part_data

    out = {'edp': res_edp, 'partition': res_partition,
           'vanilla': res_vanilla, 'peer': res_peer}

    import json
    with open(f'res_{phi_str}.json', 'w+') as f:
        json.dump(out, f, indent=4)

if __name__ == '__main__':
    main()
