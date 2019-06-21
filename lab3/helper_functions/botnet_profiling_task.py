import numpy as np
import itertools
import operator

def extract_ngrams(data, n):
    l = len(data)
    if l%n==0:
        ngrams = np.reshape(data, [l//n, n])
    else:
        ngrams = np.reshape(data[:-(l%n)], [(l-l%n)//n, n])

    return ngrams


def laplace_smoothing(unique_ngrams, ngrams_counts, max_unigram):

#     unigrams = np.unique(unique_ngrams)
    unigrams = np.arange(max_unigram+1)
#     print(unigrams)
    n = unique_ngrams.shape[1]

    comb = list(itertools.product(unigrams, repeat=n))

    smoothed_ngrams = list(unique_ngrams)
    smoothed_count = list(ngrams_counts)
    for ngram in comb:
        exists = np.sum(np.prod(np.equal(ngram, unique_ngrams), axis=1))

        if exists==0:
#             print(ngram, exists)
            smoothed_ngrams.append(ngram)
            smoothed_count.append(1)
    smoothed_ngrams = np.array(smoothed_ngrams)
    smoothed_count = np.array(smoothed_count)

    return smoothed_ngrams, smoothed_count


def extract_profile(ngrams, ngrams_counts):

    ngrams = ngrams.astype(str)
    string_cat = lambda x: ''.join(x)
    ngrams = np.apply_along_axis(string_cat, 1, ngrams)

    prof = dict(zip(ngrams, ngrams_counts))
    prof_norm = dict(zip(ngrams, ngrams_counts/sum(ngrams_counts)))

    sorted_count = sorted(prof_norm.items(), key=operator.itemgetter(0))

    return prof, prof_norm, np.array(sorted_count)


def distance(train_prof, host_prof, sorted_train_prof, sorted_host_prof, N):

    if N>=len(sorted_train_prof):
        train_ngrams = sorted_train_prof[:, 0]
    else:
        train_ngrams = sorted_train_prof[:N, 0]

    if N>=len(sorted_host_prof):
        host_ngrams = sorted_host_prof[:, 0]
    else:
        host_ngrams = sorted_host_prof[:N, 0]

    ngrams = np.append(train_ngrams, host_ngrams)
    ngrams = np.unique(ngrams)

    dist = 0
    for ngram in ngrams:
        if ngram not in train_prof.keys():
            f_train = 0
        else:
            f_train = train_prof[ngram]

        if ngram not in host_prof.keys():
            f_host = 0
        else:
            f_host = host_prof[ngram]
        dist += ((f_train-f_host) / ((f_train+f_host)/2))**2
    return dist
