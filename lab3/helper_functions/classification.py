import numpy as np
import pandas as pd
import datetime
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score


# Inspired by https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# This function plots the confusion matrix.
#
# INPUT:
#    cm        : confusion matrix to be printed (in form of a 2x2 matrix),
#    classes   : the labels corresponding to the classes (in our case 'Settled' and 'Chargeback')
#    normalize : whether to present normalized counts or not,
#    title     : title of the provided plot,
#    cmap      : the color map used for the visualization
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def performance_metrics(cm):
    accuracy = (cm[0,0] + cm[1,1])/np.sum(cm)
    precision = cm[0,0]/(cm[0,0]+cm[0,1])
    recall = cm[0,0]/(cm[0,0]+cm[1,0])
    F1_score = 2*(precision*recall)/(precision+recall)

    return accuracy, precision, recall, F1_score



def roc_analysis(X, y, clf, cv, plot_all_ROC, plot_ROC, attack=False, sm=None):
    # vectors for storing True Positives, False Negatives, False Positives and True negatives
    # respectively per cross-validation run
    num_splits = cv.get_n_splits()
    TP = np.zeros(num_splits, dtype=int)
    FN = np.zeros_like(TP)
    FP = np.zeros_like(TP)
    TN = np.zeros_like(TP)


    # Likewise for storing precision, recall and F1 score respectively extracted,
    # per cross-validation run
    PR = np.zeros(num_splits)
    R = np.zeros_like(PR)
    F1 = np.zeros_like(PR)

    # Likewise for the True Positive Rate and the Area Under Curve
    TPR = []
    AUC = []
    mean_FPR = np.linspace(0, 1, 100)

    i = 0
    if plot_ROC:
        plt.figure(figsize=(8, 10))

    # Cross-validation
    for train_index, test_index in cv.split(X, y):
        # If sm is provided, after extracting the split we first perform SMOTE on
        # training data and then proceed to training the classifier
        if sm != None:
            X_train, y_train = sm.fit_resample(X[train_index], y[train_index])
        else:
            X_train, y_train = X[train_index], y[train_index]

        # Train the classifier and extract predictions for test samples
        clf.fit(X_train, y_train)

        X_test = X[test_index]
        feat = [0,4,5]
        
        if attack:
            malicious_ind = np.where(y[test_index]==1)[0][..., np.newaxis]
            perturb = [120, 200, 2048]
            perturb = np.tile(perturb, [len(malicious_ind), 1])
            X_test[malicious_ind, feat] = X_test[malicious_ind, feat] + perturb
        
        y_pred = clf.predict(X_test)


        # Extract confusion matrix, precision, recall and F1 score for the current
        # run of cross validatin
        TN[i], FP[i], FN[i], TP[i] = confusion_matrix(y[test_index], y_pred).ravel()
        PR[i] = precision_score(y[test_index], y_pred, average='binary')
        R[i] = recall_score(y[test_index], y_pred, average='binary')
        F1[i] = f1_score(y[test_index], y_pred, average='binary')

        # Extract false positive and true positive rate correspoding to the positive (fraud) label
        fpr, tpr, thresholds = roc_curve(y[test_index], clf.predict_proba(X[test_index])[:, 1])

        # Interpolate extracted values on range of 0:100 with a step of 1
        TPR.append(np.interp(mean_FPR, fpr, tpr))

        # Set the first value of the run equal to 0
        TPR[-1][0] = 0.0

        # Extract and append to list the Area under Curve
        roc_auc = auc(fpr, tpr)
        AUC.append(roc_auc)

        # if TRUE all ROC curves correspodning to each run are plotted
        if plot_all_ROC:
            plt.plot(fpr, tpr, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i = i + 1

    # Extract mean values and standard deviations per run
    mean_TPR = np.mean(TPR, 0)
    mean_TPR[-1] = 1.0
    std_TPR = np.std(TPR, 0)
    # To assure that it doesn't go out of the graph
    upper_TPR = np.minimum(mean_TPR + std_TPR, 1)
    lower_TPR = np.maximum(mean_TPR - std_TPR, 0)

    mean_AUC = auc(mean_FPR, mean_TPR)
    std_AUC = np.std(AUC)

    mean_PR = np.mean(PR)
    mean_R = np.mean(R)
    mean_F1 = np.mean(F1)
    std_F1 = np.std(F1)

    # If true plot the mean ROC curve
    if plot_ROC:
        # Line corresponding to the 'Chance' classifier (50-50)
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

    # Extract the overall confusion matrix, by summing up the values extracted
    # from all runs.
    cm = np.array([[np.sum(TP), np.sum(FP)], [np.sum(FN), np.sum(TN)]])

    return mean_TPR, std_TPR, mean_AUC, std_AUC, mean_F1, std_F1, cm
