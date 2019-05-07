import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from PlotConfusionMatrixV2 import *

from imblearn.under_sampling import RandomUnderSampler


def roc_analysis(X, y, clf, cv, plot_all_ROC, plot_ROC, plot_cm, normalize_cm, sm = None):
    # vectors for storing True Negatives, False Positives, False-Negatives and True-Positives
    # respectively per cross-validation run
    num_splits = cv.get_n_splits()
    TP = np.zeros(num_splits, dtype = int)
    FN = np.zeros_like(TP)
    FP = np.zeros_like(TP)
    TN = np.zeros_like(TP)

    PR = np.zeros(num_splits)
    R = np.zeros_like(PR)
    F1 = np.zeros_like(PR)

    TPR = []
    AUC = []
    mean_FPR = np.linspace(0,1,100)

    # us = RandomUnderSampler(sampling_strategy = 0.9)

    i = 0
    if plot_ROC:
        plt.figure(figsize = (8, 10))
    for train_index, test_index in cv.split(X, y):
        if sm != None:
            X_train, y_train = sm.fit_resample(X[train_index], y[train_index])
            # X_train, y_train = us.fit_resample(X_train, y_train)
        else:
            X_train, y_train = X[train_index], y[train_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X[test_index])
        # print("num TP: ", np.sum(np.logical_and(y_pred == 1, y[test_index] == 1)))
        # print("num FP: ", np.sum(np.logical_and(y_pred ==1, y[test_index] == 0)))
        # print("num TN: ", np.sum(np.logical_and(y_pred == 0, y[test_index] == 0)))
        # print("num FN: ", np.sum(np.logical_and(y_pred == 0, y[test_index] == 1)))
        TN[i], FP[i], FN[i], TP[i] = confusion_matrix(y[test_index], y_pred).ravel()
        PR[i] = precision_score(y[test_index], y_pred, average = 'binary')
        R[i] = recall_score(y[test_index], y_pred, average = 'binary')
        F1[i] = f1_score(y[test_index], y_pred, average = 'binary')

        fpr, tpr, thresholds = roc_curve(y[test_index], clf.predict_proba(X[test_index])[:, 1])

        TPR.append(np.interp(mean_FPR, fpr, tpr))
        TPR[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        AUC.append(roc_auc)

        if plot_all_ROC:
            plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i = i+1

    mean_TPR = np.mean(TPR, 0)
    mean_TPR[-1] = 1.0
    std_TPR = np.std(TPR, 0)
    upper_TPR = np.minimum(mean_TPR + std_TPR, 1)
    lower_TPR = np.maximum(mean_TPR - std_TPR, 0)

    mean_AUC = auc(mean_FPR, mean_TPR)
    std_AUC = np.std(AUC)
    # print("AUC: ", mean_AUC, " +- ", std_AUC)

    mean_PR = np.mean(PR)
    mean_R = np.mean(R)
    mean_F1 = np.mean(F1)
    std_F1 = np.std(F1)
    # print("F1: ", np.mean(F1), " +- ", np.std(F1))

    if plot_ROC:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

        plt.plot(mean_FPR, mean_TPR, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_AUC, std_AUC),
             lw=2, alpha=.8)
        plt.fill_between(mean_FPR, lower_TPR, upper_TPR, color='b', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        plt.show()


    cm = np.array([np.sum(TN), np.sum(FP), np.sum(FN), np.sum(TP)])
    cm = np.reshape(cm, [2,2], order='C')
    # Plot confusion matrix with the sum of statistics gathered from crossval runs
    if plot_cm:
        plt.figure()
        plot_confusion_matrix(cm, classes = ['Non-Fraud', 'Fraud'], normalize = normalize_cm)

    return mean_TPR, std_TPR, mean_AUC, std_AUC, mean_F1, std_F1, cm
