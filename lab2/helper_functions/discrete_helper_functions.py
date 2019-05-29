import numpy as np
from saxpy.sax import sax_via_window
import operator
from matplotlib import pyplot as plt


'''
The following function computes the ngram probabilites for a given dicretization of a signal.
INPUT: sax: discretization of the signal (done via SAX with a sliding window)
OUTPUT: ngram_probs: ngram probabilities
'''
def extract_ngram_probs(sax):

    ngrams = list(sax.keys())

    l = list(sax.values())
    flat = [item for sublist in l for item in sublist]
    tot = len(flat)

    ngram_probs = {}
    for i in ngrams:
        ngram_probs[i] = len(sax[i])/tot
    return ngram_probs


'''
This function extracts a training profile with the most frequent ngrams extracted from
a training dataset.
INPUT: train_probs: probabilites of ngrams extracted from the training dataset
       thres:
        - if thres = 1: all the ngrams are incorporated
        - if 0 < thres < 1, keep all the ngrams with correspoding probability >= thres
        - if thres>1 (e.g. 100), keep the top thres (e.g. 100) most frequent ngrams
OUTPUT: train_prof: training profile; list with the most frequent ngrams
'''
def extract_train_prof(train_probs, thres=None):
    train_ngrams = list(train_probs.keys())

    train_prof = []
    if thres == 1:
        train_prof = sorted(train_probs.keys(), key=operator.itemgetter(1), reverse=True)
    elif thres > 1:
        sorted_probs = sorted(train_probs.keys(), key=operator.itemgetter(1), reverse=True)
        train_prof = sorted_probs[:thres]
    elif thres >0 and thres < 1:
        for i in train_ngrams:
            if train_probs[i] >= thres:
                train_prof.append(i)
    else:
        print("Give correct threshold")

    return train_prof


'''
The function extracts the indices (sample index) from a test set where an alarm has been raised.
The ngrams encountered, not present in the extracted training prfile, leads to alarm raising.
INPUT: test_sax: discretization of a test signal,
       train_prof: training profile
OUTPUT: alarm_regions: mask indicating the indices where an alarm has been raised
'''
def extract_alarm_regions(test_sax, train_prof):
    test_ngrams = list(test_sax.keys())
    alarm_ngrams = [ngram for ngram in test_ngrams if ngram not in train_prof]
    alarm_regions = []
    for i in alarm_ngrams:
        alarm_regions.append(test_sax[i])
    alarm_regions = [x for sublist in alarm_regions for x in sublist]
    return alarm_regions
