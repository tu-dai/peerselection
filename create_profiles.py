import gzip
import json
import pickle
import time

import numpy as np
from tqdm import tqdm

from mallow import generate_mallows


def create_profile(num_items, phi):
    return {i: ranks.tolist() for i, ranks in enumerate(generate_mallows(num_items, num_items, phi))}


if __name__ == '__main__':

    num_exps = 10000

    num_items = 200
    items = np.arange(0, num_items)
    scores = np.arange(0, num_items)

    k = 50

    phi = 0.8

    for phi in [0.2, 0.5, 0.8]:
        profiles = [create_profile(num_items, phi) for _ in tqdm(range(num_exps))]

        # json_str = json.dumps(profiles)
        # json_bytes = json_str.encode('utf-8')
        t = time.time()
        with gzip.open(f'profiles_0{str(phi)[-1]}.gzip', 'w', compresslevel=9) as f:
            pickle.dump(profiles, f, protocol=pickle.HIGHEST_PROTOCOL)
            # f.write(profiles)
        print(f'gzip took {time.time() - t} secs')
