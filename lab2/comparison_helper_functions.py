def detect_num_attacks(test_labels):
    na=0
    pos=test_labels[0]
    for y in test_labels[1:]:
        if y==1:
            if pos==0:
                pos=1
        else:
            if pos==1:
                pos=0
                na=na+1
    return na

def compute_sttd(test_labels, predicted_labels):
    num_attacks = detect_num_attacks(test_labels)
    total_auc = np.sum(test_labels==1)
    intersect =  np.logical_and(test_labels==1, predicted_labels==1)
    intersect_auc = np.sum(intersect)
    diff = total_auc - intersect_auc
    sttd = 1 - num_attacks * diff
    return sttd

def estimate_confusion_matrix(test_labels, alarm_ind):
    TP = np.sum(np.isin(alarm_ind, test_labels[test_labels==1]))
    FP = np.sum(TP[TP==0])
    FN = np.sum(np.isin(test_labels[test_labels==1], alarm_ind, invert=True))
    TN = np.sum(np.isin(test_labels[test_labels==0], alarm_ind, invert=True))

    cm = np.array([[TP, FP][FN, TN]])
    return cm

def compute_scm(cm):
    TPR = cm[0,0]/(cm[0,0] + cm[1,0])
    TNR = cm[1,1]/(cm[0,1] + cm[1,1])

    scm = (TPR + TNR)/2

def compute_s(test_labels, predicted_labels, cm, gamma):
    sttd = compute_sttd(test_labels, predicted_labels)
    scm = compute_scm(cm)
    s = gamma*sttd + (1-gamma)*scm
    return s
