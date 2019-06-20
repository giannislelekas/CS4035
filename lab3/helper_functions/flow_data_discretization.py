import numpy as np


# def discretize_ordinal(val, ordinal_val):
#     if len(val)==1:
#         disc = val>ordinal_val
#     else:
#         disc = []
#         for v in val:
#             for i in range(len(ordinal_val)):
#                 if v<=ordinal_val[i]:
#                     disc.append(i)
#                     break
#                 if i==len(ordinal_val)-1:
#                     disc.append(i+1)
#     return np.array(disc)

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


def extract_code(values):
    M = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        M[j] = len(np.unique(values[:,j]))
    spacesize = np.prod(M)

    code = np.zeros(values.shape[0])
    for i in range(values.shape[0]):
        s = spacesize
        for j in range(values.shape[1]):
            code[i] = code[i] + values[i,j] * s/M[j]
            s = s / M[j]
    return code.astype(int), M
