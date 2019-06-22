import numpy as np
import hashlib
import mmh3

'''
This function performs the Count-Min sketch, for given values of depth and width.
As hash functions MurmuHash has been used (https://pypi.org/project/mmh3/).
INPUT: data: data for the play (in this case hosts),
       depth, width: parameters for constructing the CM matrix
OUTPUT: cm: the extracted matrix,
        seeds: random seeds used for hashing (necessary for looking back to extract
                counts in the following function estimate_counts)
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
This function computes the estimated counts for each unique host, based on the
extracted CM matrix.
INPUT: unique_data: unique_hosts,
       cm: CM matrix
       seeds: seeds used for hashing
OUTPUT: estim: estimated counts
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


'''
This fuction evaluates a play, based on a given ground truth. Recall value is
computed based on the ratio of ground truth IPs occuring also in the play. Additionally,
the distance-divergence of computed frequencies are computed, for each IP of
ground-truth occuring also at the play results we compute the absolute difference
of frequencies; in case the IP is missing from the play only the respective ground-truth
frequency is added.
INPUT: ground_truth top-10 dataframe,
       df: play top-10
OUTPUT: rec: recall,
        dis: distance
'''
def evaluate(ground_truth, df):

    n = len(ground_truth)
    ground_truth_hosts = list(ground_truth.index)
    df_hosts = list(df.index)

    rec = np.sum(np.isin(df_hosts, ground_truth_hosts))/n

    dis = 0
    for i in ground_truth_hosts:
        if i in df_hosts:
            dis += abs(ground_truth[i] - df.loc[i].values[0])
        else:
            dis += ground_truth[i]

    return rec, dis
