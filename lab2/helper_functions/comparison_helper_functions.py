import numpy as np
from matplotlib import pyplot as plt


'''
This function computes the number of attacks present in a set based on the number of labels
transitions (each discrete attack event is a sequence of labels 0->1...1->0).
INPUT: test_labels: the labels of the to be-examined set
OUTPUT: num_attacks: number of attacks detected,
        attack_start, attack_duration: lists with the indices of the start of each attack
                                       and the attack duration respectively
'''
def attack_detection(test_labels):
    num_attacks=0
    attack_duration = []
    attack_start = []
    pos=test_labels[0]
    dur = 0
    ind = 1

    if pos==1:
        dur = 1
        attack_start.append(ind)

    for y in test_labels[1:]:
        ind = ind+1
        if y==1:
            dur = dur + 1
            if pos==0:
                pos=1
                attack_start.append(ind)
        else:
            if pos==1:
                pos=0
                num_attacks=num_attacks+1
                attack_duration.append(dur)
                dur = 0
    return num_attacks, attack_start, attack_duration


'''
This function estimates the time when an attack was detected. The time-to-detect ttd is
an integer in range [0, attack_duration[i]], for each attack event i. In case an attack
is not detected ttd = attack_duration[i].
INPUT: test_labels, predicted labels: true, predicted labels of the set respectively
OUTPUT: num_attacks: number of attacks detected,
        attack_start, attack_duration: lists with the indices of the start of each attack
                                       and the attack duration respectively
        attack_detected: list with the ttd for each attack event
'''
def times_to_detect(test_labels, predicted_labels):
    num_attacks, attack_start, attack_duration = attack_detection(test_labels)

    attack_detected = []
    for i in range(num_attacks):
        # Examine the predictions within the range of each attack and extract the first 1
        predictions = predicted_labels[attack_start[i]:attack_start[i] + attack_duration[i]]
        # This returns the first occurence of predictions==1
        ind = np.argmax(predictions)
        if predictions[ind] == 1:
            attack_detected.append(ind)
        else:
            attack_detected.append(attack_duration[i])
    return num_attacks, attack_start, attack_duration, attack_detected


'''
This function computes the score Sttd.
INPUT: test_labels, predicted_labels: true, predicted labels of the set respectively
OUTPUT: sttd: score
'''
def compute_sttd(test_labels, predicted_labels):
    num_attacks, attack_start, attack_duration, attack_detected = times_to_detect(test_labels, predicted_labels)
    sttd = 1 - (1/num_attacks)*np.sum(np.divide(attack_detected, attack_duration))

    return sttd


'''
This function estimates the confusion matrix based on the extracted predictions.
INPUT: test_labels, predicted_labels: true, predicted labels of the set respectively
OUTPUT: cm: confusion matrix
'''
def estimate_confusion_matrix(test_labels, predicted_labels):

    TP = np.sum(np.logical_and(predicted_labels==1, test_labels==1))
    FP = np.sum(np.logical_and(predicted_labels==1, test_labels==0))
    TN = np.sum(np.logical_and(predicted_labels==0, test_labels==0))
    FN = np.sum(np.logical_and(predicted_labels==0, test_labels==1))

    cm = np.array([[TP, FP], [FN, TN]])

    return cm


'''
This function estimates accuracy, precision, recall and F1-score, based on the
extracted confusion matrix.
INPUT: cm: confusion matrix
OUTPUT: accuracy, precision, recall, F1-score: metrics
'''
def performance_metrics(cm):
    accuracy = (cm[0,0] + cm[1,1])/np.sum(cm)
    precision = cm[0,0]/(cm[0,0]+cm[0,1])
    recall = cm[0,0]/(cm[0,0]+cm[1,0])
    F1_score = 2*(precision*recall)/(precision+recall)

    return accuracy, precision, recall, F1_score



'''
This function estimates the score derived from the confusion matrix.
INPUT: cm: confusion matrix
OUTPUT: scm: score
'''
def compute_scm(cm):
    TPR = cm[0,0]/(cm[0,0] + cm[1,0])
    TNR = cm[1,1]/(cm[0,1] + cm[1,1])

    scm = (TPR + TNR)/2
    return scm


'''
This function estimates the overall score based on the extracted scores sttd and
scm. gamma (1-gamma) is the wheight assigned to sttd (scm).
INPUT: test_labels, predicted_labels: true, predicted labels of the set respectively,
       gamma: weight
OUTPUT: s: score
'''
def compute_s(test_labels, predicted_labels, gamma):
    cm = estimate_confusion_matrix(test_labels, predicted_labels)
    sttd = compute_sttd(test_labels, predicted_labels)
    scm = compute_scm(cm)
    s = gamma*sttd + (1-gamma)*scm

    return s



'''
This function provides a graph for the true and predicted labels for evaluation purposes.
INPUT: test_labels, predicted_labels: true, predicted labels of the set respectively
'''
def plot(test_labels, predicted_labels):
    plt.figure(figsize=(20,5))

    alarm_ind = np.where(predicted_labels)
    intersect = np.isin(alarm_ind, np.where(test_labels))
    ind = np.array(alarm_ind)[intersect]
    Y = np.zeros_like(test_labels)
    Y[ind] = 1

    plt.plot(predicted_labels, color = 'r', label='predicted', alpha=0.4)
    plt.plot(test_labels, color = 'g', label='actual')
    plt.fill(Y, facecolor='g', alpha=1)

    plt.xlabel("sample index")
    plt.ylabel("Label y")
    plt.legend()
