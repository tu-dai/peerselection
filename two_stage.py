import argparse
import gzip
import json
import pickle
import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from create_profiles import create_profile
from peerselect.peerselect import impartial, profile_generator
from run_from_file import calc_metrics, get_probs
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_two_stage(items, profile, clusters, scores, initial_reviews, max_reviews,
                  agg_method, k, top_freeze=1, bottom_freeze=1, normalize=False):
    score_matrix = profile_generator.profile_classes_to_score_matrix(profile, scores)

    init_assignment = profile_generator.generate_approx_m_regular_assignment(items, initial_reviews,
                                                                             clusters, randomize=True)

    new_scores = profile_generator.restrict_score_matrix(score_matrix, init_assignment)

    if initial_reviews == max_reviews:
        return agg_method(new_scores, k, partition=clusters, normalize=normalize)

    if top_freeze > 0:
        top_winners = agg_method(new_scores, top_freeze, partition=clusters, normalize=normalize)
    else:
        top_winners = []

    top_losers = agg_method(score_matrix.max() - new_scores, bottom_freeze, partition=clusters, normalize=normalize)
    removed = top_winners + top_losers

    counts = {item: initial_reviews for item in items if item not in removed}

    new_assignment = profile_generator.generate_approx_m_regular_assignment(items, max_reviews - top_freeze,
                                                                            clusters, randomize=True,
                                                                            review_counts=counts,
                                                                            agent_assignment=init_assignment)
    new_scores = profile_generator.restrict_score_matrix(score_matrix, new_assignment)
    new_scores[removed] = 0

    indices, cur_count = np.unique(np.array(sum(new_assignment.values(), [])), return_counts=True)
    assignment_count = np.zeros(items.shape[0])
    assignment_count[indices]  = cur_count

    new_winners = agg_method(new_scores, k - top_freeze, partition=clusters, normalize=normalize,
                             assignment_count=assignment_count)
    outcome = top_winners + new_winners

    return outcome


def run_instance(items, profile, clusters, scores, initial_reviews, max_reviews,
                 k, top_freeze=1, bottom_freeze=1):

    agg_method = impartial.vanilla
    # print("running vanilla")
    vanilla1 = run_two_stage(items, profile, {}, scores, max_reviews, max_reviews,
                             agg_method, k)
    vanilla2 = run_two_stage(items, profile, {}, scores, initial_reviews, max_reviews,
                             agg_method, k, top_freeze, bottom_freeze, False)

    agg_method = impartial.partition_explicit
    # print("running partition")
    partition1 = run_two_stage(items, profile, clusters, scores, max_reviews, max_reviews,
                               agg_method, k)
    partition2 = run_two_stage(items, profile, clusters, scores, initial_reviews, max_reviews,
                               agg_method, k, top_freeze, bottom_freeze, False)

    agg_method = impartial.exact_dollar_partition_explicit
    # print("running edp")
    edp1 = run_two_stage(items, profile, clusters, scores, max_reviews, max_reviews,
                         agg_method, k, normalize=True)
    edp2 = run_two_stage(items, profile, clusters, scores, initial_reviews, max_reviews,
                         agg_method, k, top_freeze, bottom_freeze, True)

    # print("finished")
    return vanilla1, vanilla2, partition1, partition2, edp1, edp2



def main():
    num_exps = 100
    num_items = 200
    num_clusters = 3
    items = np.arange(0, num_items)
    scores = np.arange(0, num_items)

    parser = argparse.ArgumentParser(description="two stage parameters.")

    parser.add_argument("phi", type=float, help="mallow noise.")
    parser.add_argument("k", type=int, help="Number of items to select.")
    parser.add_argument("num_reviews", type=int, help="Number of reviews reviewers.")
    parser.add_argument("first_round_reviews", type=int, help="Number of reviews in the first round.")
    parser.add_argument("top_freeze", type=int, help="Number of top items to freeze after the first round.")
    parser.add_argument("bottom_freeze", type=int, help="Number of bottom items to remove after the first round.")

    args = parser.parse_args()

    k = args.k

    phi = args.phi
    phi_str = str(phi).replace(".", "")

    num_reviews = args.num_reviews
    first_round_reviews = args.first_round_reviews
    top_freeze = args.top_freeze
    bottom_freeze = args.bottom_freeze

    print("loading profiles")
    with gzip.open(f'profiles_{phi_str}.gzip', 'rb') as f:
        profiles = pickle.load(f)
        profiles = profiles[:num_exps]

    with gzip.open(f'partition_{num_clusters}.gzip', 'rb') as f:
        part_data = pickle.load(f)
        clusters = part_data['clusters']
        del part_data


    metrics = {'precision@k': [], 'positive_borda': [], 'negative_borda': []}
    vanillas1 = {'outcomes': [], 'metrics': deepcopy(metrics)}
    vanillas2 = {'outcomes': [], 'metrics': deepcopy(metrics)}
    partitions1 = {'outcomes': [], 'metrics': deepcopy(metrics)}
    partitions2 = {'outcomes': [], 'metrics': deepcopy(metrics)}
    edps1 = {'outcomes': [], 'metrics': deepcopy(metrics)}
    edps2 = {'outcomes': [], 'metrics': deepcopy(metrics)}
    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(num_exps):
            profile = profiles[i]
            clustering = clusters[i]
            # run_instance( items, profile, clustering, scores,
            # first_round_reviews, num_reviews, k, top_freeze,
            # bottom_freeze)
            futures.append(executor.submit(run_instance, items, profile, clustering, scores,
                                           first_round_reviews, num_reviews, k, top_freeze,
                                           bottom_freeze))

        for future in tqdm(as_completed(futures), total=num_exps, desc='Experiments'):
            vanilla1, vanilla2, partition1, partition2, edp1, edp2 = future.result()
            vanillas1['outcomes'].append(vanilla1)
            vanillas2['outcomes'].append(vanilla2)
            partitions1['outcomes'].append(partition1)
            partitions2['outcomes'].append(partition2)
            edps1['outcomes'].append(edp1)
            edps2['outcomes'].append(edp2)

            for outcome, mdict in zip([vanilla1, vanilla2, partition1, partition2, edp1, edp2],
                                      [vanillas1, vanillas2, partitions1, partitions2, edps1, edps2]):
                prec, pb, nb = calc_metrics(outcome, num_items, k)
                mdict['metrics']['precision@k'].append(prec)
                mdict['metrics']['positive_borda'].append(pb)
                mdict['metrics']['negative_borda'].append(nb)

    res = {'v1': {}, 'v2': {}, 'p1': {}, 'p2': {}, 'e1': {}, 'e2': {}}
    res['v1']['probs'] = get_probs(vanillas1['outcomes'], num_exps, num_items).tolist()
    res['v2']['probs'] = get_probs(vanillas2['outcomes'], num_exps, num_items).tolist()
    res['p1']['probs'] = get_probs(partitions1['outcomes'], num_exps, num_items).tolist()
    res['p2']['probs'] = get_probs(partitions2['outcomes'], num_exps, num_items).tolist()
    res['e1']['probs'] = get_probs(edps1['outcomes'], num_exps, num_items).tolist()
    res['e2']['probs'] = get_probs(edps2['outcomes'], num_exps, num_items).tolist()

    for method, mdict in zip(['v1', 'v2', 'p1', 'p2', 'e1', 'e2'],
                        [vanillas1, vanillas2, partitions1, partitions2, edps1, edps2]):
        res[method]['metrics'] = {}
        for metric_name in ['precision@k', 'positive_borda', 'negative_borda']:
            res[method]['metrics'][metric_name] = {'mean': np.mean(mdict['metrics'][metric_name]),
                                                   'std': np.std(mdict['metrics'][metric_name])}

    # with open(f'res_{phi_str}.json', 'w+') as f:
    fn = f'two_{phi_str}_{k}_{num_reviews}_{first_round_reviews}_{top_freeze}_{bottom_freeze}.json'
    with open(fn, 'w+') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()