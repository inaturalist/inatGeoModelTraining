import math

import numpy as np

def make_rand_samples(batch_size):
    rand_loc = np.random.uniform(size=(batch_size, 2))

    theta1 = 2.0*math.pi*rand_loc[:, 0]
    theta2 = np.arccos(2.0*rand_loc[:, 1] - 1.0)

    lat = 1.0 - 2.0*theta2/math.pi
    lng = (theta1/math.pi) - 1.0

    return np.array(list(zip(lng, lat)))

def get_idx_subsample_observations(labels, hard_cap=-1):
    if hard_cap == -1:
        return np.arange(len(labels))

    print(f"  subsampling (up to) {hard_cap} per class for the training set")
    class_counts = {id: 0 for id in np.unique(labels)}
    ss_rng = np.random.default_rng()
    idx_rand = ss_rng.permutation(len(labels))
    idx_ss = []
    for i in idx_rand:
        class_id = labels[i]
        if class_counts[class_id] < hard_cap:
            idx_ss.append(i)
            class_counts[class_id] += 1
    idx_ss = np.sort(idx_ss)
    print(f"  final training set size: {len(idx_ss)}")
    return idx_ss
