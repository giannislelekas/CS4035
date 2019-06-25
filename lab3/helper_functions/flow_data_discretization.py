import numpy as np


'''
This function performs ordinal discretization as done in https://ieeexplore.ieee.org/abstract/document/7987293.
INPUT: val: values of the feature to be discretized,
       ranks: number of ordinal ranks-percentiles used for the discretization
       (e.g. 5 ranks, the 20,40,60,80 percentiles are considered, for value <=20 -> 0, 20<val<=40 -> 1, etc)
OUTPUT: disc: discretized feature
'''
def discretize_ordinal(val, ranks):

    ordinal_ranks = np.linspace(0, 1, ranks, endpoint=False)
    ordinal_ind = np.ceil(ordinal_ranks[1:] * len(val))
    ordinal_val = np.array(sorted(val))[ordinal_ind.astype(int)]

    if len(ordinal_val)==1:
        disc = (val>ordinal_val).astype(int)
    else:
        disc = []
        for v in val:
            for i in range(len(ordinal_val)):
                if v<=ordinal_val[i]:
                    disc.append(i)
                    break
                if i==len(ordinal_val)-1:
                    disc.append(i+1)
    return np.array(disc)


'''
This function extract code values as done in https://ieeexplore.ieee.org/abstract/document/7987293, combining
a set of features into a single value.
INPUT: values: teh feature values from which codes will be extracted.
OUTPUT: code: extracted code values,
        M: spacesize (number of unique values) for each feature
'''
def extract_code(values):
    M = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        M[j] = len(np.unique(values[:,j]))
    spacesize = np.prod(M)

    code = np.zeros(values.shape[0], dtype=int)
    for i in range(values.shape[0]):
        s = spacesize
        for j in range(values.shape[1]):
            code[i] = code[i] + values[i,j] * s/M[j]
            s = s / M[j]
    return code, M
