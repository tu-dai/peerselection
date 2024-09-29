import gzip
import json
import pickle
import random
import time

import numpy as np
from tqdm import tqdm

from create_edp import create_profile
from peerselect.peerselect import impartial, profile_generator

num_exps = 10000

num_items = 200
items = np.arange(0, num_items)
scores = np.arange(0, num_items)

k = 50

for num_clusters in tqdm([3], desc='#clusters', leave=False, position=0):
    clusters = [impartial.even_partition_order(sorted(items, key=lambda j: random.random()),
                                               num_clusters) for _ in range(num_exps)]

    # test = profile_generator.generate_approx_m_regular_assignment(items, 3,
    #                                                               clusters[0],
    #                                                               randomize=True)
    #
    # counts = {1:0,5:0,10:0,15:0,20:0}
    # test2 = profile_generator.generate_approx_m_regular_assignment(items, 1,
    #                                                                clusters[0],
    #                                                                randomize=True,
    #                                                                review_counts=counts,
    #                                                                agent_assignment=test)

    assignments_1 = [profile_generator.generate_approx_m_regular_assignment(items, 3,
                                                                            clusters[i],
                                                                            randomize=True) for
                     i in range(num_exps)]

    assignments_20 = [profile_generator.generate_approx_m_regular_assignment(items, 20,
                                                                             clusters[i],
                                                                             randomize=True) for
                      i in range(num_exps)]

    data = {'clusters': clusters,
            'assignments_1': assignments_1,
            'assignments_20': assignments_20}

    t = time.time()
    with gzip.open(f'partition_{num_clusters}.gzip', 'w', compresslevel=9) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        # f.write(profiles)
    print(f'gzip took {time.time() - t} secs')



