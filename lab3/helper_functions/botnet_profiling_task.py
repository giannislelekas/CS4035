import numpy as np
import itertools
import operator


'''
This function extract n-grams based on a sliding window of a given length.
INPUT: data: the values to extract n-grams from,
       n: length of the sliding window
OUTPUT: ngrams: extracted n-grams
'''
def extract_ngrams(data, n):
    l = len(data)
    ngrams = np.zeros((l-n+1,n))
    for i in range(l-n+1):
        ngrams[i, :] = data[i:i+n]

    return ngrams


'''
This function implements laplace smoothing, essentially adding combinations not present
in a given profile. For the generation of all possible combinations the cartesian product is used.
INPUT: unique_ngrams: given set of n-grams of a profile,
       ngrams_counts: respective frequency of the n-grams
       max_unigram: the max value encountered, used to generate all the possible combinations
OUTPUT: smoothed_ngrams, smoothed_count: entire list of combinations with the added n-grams, appended
                                        at the end with a count of 1.
'''
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


'''
This function extracts a profile given a set of n-grams and their respective frequency.
Essentially creates a dictionary for the profile with n-grams and their normalized frequency.
INPUT: ngrams, ngrams_counts: n-grams and respective frequency
OUTPUT: prof: profile with respective frequency,
        prof_norm: profile with respective normalized frequency
        sorted_count: sorted profile based on the normalized frequency
'''
def extract_profile(ngrams, ngrams_counts):

    ngrams = ngrams.astype(str)
    string_cat = lambda x: ''.join(x)
    ngrams = np.apply_along_axis(string_cat, 1, ngrams)

    prof = dict(zip(ngrams, ngrams_counts))
    prof_norm = dict(zip(ngrams, ngrams_counts/sum(ngrams_counts)))

    sorted_count = sorted(prof_norm.items(), key=operator.itemgetter(0))

    return prof, prof_norm, np.array(sorted_count)


'''
This function implements the distance metric of the CNG method found in
https://web.cs.dal.ca/~vlado/papers/pst04.pdf.
INPUT: train_prof, host_prof: training and host profiles respectively,
       sorted_train_prof, sorted_host_prof: respective sorted profiles based on
                                            normalized frequency
OUTPUT: dist: extracted distance
'''
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



def performance_metrics(cm):
    accuracy = (cm[0,0] + cm[1,1])/np.sum(cm)
    precision = cm[0,0]/(cm[0,0]+cm[0,1])
    recall = cm[0,0]/(cm[0,0]+cm[1,0])
    F1_score = 2*(precision*recall)/(precision+recall)

    return accuracy, precision, recall, F1_score
