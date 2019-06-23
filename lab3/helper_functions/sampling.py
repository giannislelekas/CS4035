import numpy as np

'''
Deprecated. We use the next function.
'''
# def sample(data, k):
#     reservoir = []
#     for t, item in enumerate(data):
#         if t < k:
#             reservoir.append(item)
#         else:
#             m = np.random.rand()
#             if m <= k/t:
#                 ind = np.random.randint(0,k)
#                 reservoir[ind] = item
#     return reservoir


'''
This function performs the RESERVOIR sampling. Inspired by
https://en.wikipedia.org/wiki/Reservoir_sampling.
INPUT: data: the hosts communicating with the infected host,
       k: the size of the reservoir
OUTPUT: reservoir: resulting sampling
'''
def sample(data, k):
    reservoir = []
    for t, item in enumerate(data):
        if t < k:
            reservoir.append(item)
        else:
            # t+1 to include t in the range
            m = np.random.randint(0, t+1)
            if m < k:
                reservoir[m] = item
    return reservoir


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
