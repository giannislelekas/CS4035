import numpy as np
from saxpy.sax import sax_via_window
import operator
from tslearn.piecewise import SymbolicAggregateApproximation
from matplotlib import pyplot as plt
from helper_functions.comparison_helper_functions import *


'''
This function extracts the optimal parameters for the discretization to be used, based on the
maximum score S (refer to the comparison task in the report, for the definition of S).
INPUT: train_signal, train_labels, test_signal, test_labels: time-series and labels of the training and test signals,
       window_size: the length of the sliding window,
       paa_segments: the number of segments used for Piecewise Aggregate Approximation,
       vocab_size: number of letters used for the discretization (number of quantized states)
       gamma: multiplier for components of score S
OUTPUT: S: array with extracted scores for every examined case,
        optimal_labels: predicted labels for the case with max score S

'''
def optimal_parameters(train_signal, train_labels, test_signal, test_labels, window_size, paa_segments, vocab_size, gamma):
    S = np.zeros((len(window_size), len(paa_segments), len(vocab_size)))

    max_S = 0
    optimal_labels = np.zeros_like(test_labels)

    for w in range(len(window_size)):
        for p in range(len(paa_segments)):
            for v in range(len(vocab_size)):
                train_sax=sax_via_window(train_signal, window_size[w], paa_segments[p], vocab_size[v],
                   nr_strategy='none', z_threshold=0.01)
                test_sax = sax_via_window(test_signal, window_size[w], paa_segments[p], vocab_size[v],
                             nr_strategy='none')

                train_probs = extract_ngram_probs(train_sax)
                train_prof = extract_train_prof(train_probs, thres=1)
                alarm_regions = extract_alarm_regions(test_sax, train_prof)

#                 print("Attacks detected: ", len(alarm_regions))
                if not alarm_regions:
                    continue

                predicted_labels = np.zeros_like(test_labels)
                predicted_labels[alarm_regions] = 1

                s = compute_s(test_labels, predicted_labels, gamma)
                if s > max_S:
                    optimal_labels = predicted_labels
                S[w,p,v] = s

        return S, optimal_labels


'''
This function computes the discretization of the signal, based on the extracted parameters
from  function.
INPUT: raw_signal: the signal to be discretized,
       window_size: the length of the sliding window,
       paa_segments: the number of segments used for Piecewise Aggregate Approximation,
       alphabet_size: number of letters used for the discretization (number of quantized states)
OUTPUT: discrete_signal: the discretized signal
'''
def discretize(raw_signal, window_size, paa_segments, alphabet_size):
    sax = SymbolicAggregateApproximation(n_segments=paa_segments, alphabet_size_avg=alphabet_size)
    discrete_signal = []
    num = len(raw_signal)//window_size

    for i in range(num):
        raw_data = raw_signal[i*window_size : (i+1)*window_size]
        disc = sax.inverse_transform(sax.fit_transform(raw_data))
        discrete_signal.append(np.squeeze(disc))
    discrete_signal = [x for sublist in discrete_signal for x in sublist]

    return discrete_signal



def discretizeV2(raw_signal, window_size, paa_segments):
    discrete_signal = []
    num = window_size//paa_segments

    for i in range(0, len(signal), window_size):
        for ngram in list(train_sax.keys()):
            if i in train_sax[ngram]:
                for n in range(8):
                    discrete_signal.append(np.tile(ngram[n], num))

    discrete_signal = [x for sublist in discrete_signal for x in sublist]
    return discrete_signal



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
