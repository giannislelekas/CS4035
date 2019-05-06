import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from PlotConfusionMatrixV2 import *


def roc_analysis(X, y, clf, cv, plot_all, normalize_cm):
    # vectors for storing True Negatives, False Positives, False-Negatives and True-Positives
    # respectively per cross-validation run
    num_splits = cv.get_n_splits()
    TP = np.zeros(num_splits, dtype = int)
    FN = np.zeros_like(TP)
    FP = np.zeros_like(TP)
    TN = np.zeros_like(TP)

    TPR = []
    AUC = []
    mean_FPR = np.linspace(0,1,100)

    i = 0
    plt.figure(figsize = (8, 10))
    for train_index, test_index in cv.split(X, y):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        TN[i], FP[i], FN[i], TP[i] = confusion_matrix(y[test_index], y_pred).ravel()
        print(TN[i], FP[i], FN[i], TP[i])
        fpr, tpr, thresholds = roc_curve(y[test_index], clf.predict_proba(X[test_index])[:, 1])

        TPR.append(np.interp(mean_FPR, fpr, tpr))
        TPR[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        AUC.append(roc_auc)

        if plot_all:
            plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i = i+1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_TPR = np.mean(TPR, 0)
    mean_TPR[-1] = 1.0
    std_TPR = np.std(TPR, 0)
    upper_TPR = np.minimum(mean_TPR + std_TPR, 1)
    lower_TPR = np.maximum(mean_TPR - std_TPR, 0)

    mean_AUC = auc(mean_FPR, mean_TPR)
    std_AUC = np.std(AUC)

    plt.plot(mean_FPR, mean_TPR, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_AUC, std_AUC),
         lw=2, alpha=.8)
    plt.fill_between(mean_FPR, lower_TPR, upper_TPR, color='b', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # Plot confusion matrix with the sum of statistics gathered from crossval runs
    plt.figure()
    cm = np.array([np.sum(TN), np.sum(FP), np.sum(FN), np.sum(TP)])
    cm = np.reshape(cm, [2,2], order='C')
    plot_confusion_matrix(cm, classes = ['Non-Fraud', 'Fraud'], normalize = normalize_cm)

    return mean_TPR, std_TPR, mean_AUC, std_AUC, cm
