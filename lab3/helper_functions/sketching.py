import numpy as np
import hashlib
import mmh3

'''
'''
def construct_cm(data, depth, width):

    cm = np.zeros((depth, width), dtype=int)
    seeds = np.random.choice(10*width, depth, replace=False)

    for host in data:
        for d in range(depth):
            hash_index = mmh3.hash(host, seeds[d], signed=False) % width
            cm[d, hash_index] = cm[d, hash_index] + 1

    return cm, seeds


'''
'''
def estimate_counts(unique_data, cm, seeds):

    depth, width = cm.shape

    estim = np.ones(len(unique_data), dtype=int) * np.max(cm)
    i = 0
    for host in unique_data:
        for d in range(depth):
            hash_index = mmh3.hash(host, seeds[d], signed=False) % width
            if cm[d, hash_index] < estim[i]:
                estim[i] = cm[d, hash_index]

        i=i+1

    return estim
