import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import decomposition


def apply_pca(X, n):
    normalize = StandardScaler()
    norm_data = normalize.fit_transform(X)
    print(norm_data.shape)
    pca = PCA(n_components= X.shape[1])
    data = pca.fit_transform(norm_data)
    print(data.shape)
    print('{:0.0%} of variance is captured'.format(sum(pca.explained_variance_ratio_)))

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.show()
    reconstruct = pca.inverse_transform(data)

    return data, reconstruct, pca


'''
This function ....
INPUT: ....
OUTPUT
'''
def determine_num_components(X, threshold):
    pca = PCA()
    pca.fit(X)

    i = 0
    for comp in pca.components_:
        # print("Comp shape: ", comp.shape)
        proj = np.matmul(comp, X.T)
        # print("Proj shape: ", proj.shape)
        std = np.std(proj)
        mean = np.mean(proj)
        print('mean: ', mean)
        print("std: ", std)
        div = np.max(np.abs(proj - mean))
        print("max :", np.max(np.abs(proj)))
        print("max div: ", div)
        if div > threshold * std:
            break
        i = i + 1
    P_normal = pca.components_[:i, :].T
    P_anomal = pca.components_[i:, :].T
    return i, P_normal, P_anomal


'''
This function ....
INPUT: ....
OUTPUT
'''
def det_anomalies(data, data_test, num,threshold):

    # to eigenvectors prepei na proerxetai apo kei pou epilegei twn arithmo n
    model= PCA(n_components= num)
    model.fit(data)
    eigenvectors = model.components_
    print("comp shape: ", eigenvectors.shape)
    # Identity martrix
    I = np.eye(data.shape[1])
    # P matrix of m x r where r is the number of normal axes and m the number of components
    P = np.transpose(eigenvectors[0: num-1])
    P_T = np.transpose(P) # P Transpose matrix
    C = np.matmul(P, P_T) # C= P x P_T
    y_bar = np.zeros(data_test.shape)
    SPE = np.zeros(data_test.shape[0])    # squared prediction error
    predicted_labels= np.zeros(data_test.shape[0])
    for i in range(data_test.shape[0]):
        y = data_test[i]
        y_bar[i]= np.matmul(I - C, y)
        SPE[i]= np.square(np.linalg.norm(y_bar[i], 2))
        if SPE[i] > threshold:
            predicted_labels[i]=1

    return  predicted_labels

'''
This function ....
INPUT: ....
OUTPUT
'''
def residuals(data,P_n,P_an):

    y_hat = np.zeros_like(data)
    y_bar = np.zeros_like(data)
    for i in range(len(data)):
        y_hat[i, :] = np.matmul(np.matmul(P_n, P_n.T), data[i, :])
        y_bar[i, :] = np.matmul(np.matmul(P_an, P_an.T), data[i, :])
    residual = np.sum(np.square(y_bar), axis=1)
    return residual


def compute_residuals(data, rec):

    residuals = data - rec
    residuals = np.sum(np.square(residuals),axis=1)
    return residuals


def pca_data(X, n):

    normalize = StandardScaler()
    norm_data = normalize.fit_transform(X)
    #print(norm_data.shape)
    pca = PCA(n_components = n)
    data = pca.fit_transform(norm_data)
    #print(data.shape)
    #print('{:0.0%} of variance is captured'.format(sum(pca.explained_variance_ratio_)))
    reconstructed_data = pca.inverse_transform(data)

    return data, reconstructed_data, pca


'''
This function ....
INPUT: ....
OUTPUT
'''
def Q_stat(pca, num):

    variance = pca.explained_variance_
    print("Variance: ", variance)
    # sort = np.sort(variance)
    # rev_sort = sort[-1::-1]
    l1 = variance[num+1:]
    l2 = np.power(l1, 2)
    l3 = np.power(l1, 3)
    F1 = sum(l1)
    F2 = sum(l2)
    F3 = sum(l3)
    # a =
    Ca = 0.999  # number they used in the paper
    h0 = 1 - 2.0 * F1 * F3 / (3.0 * (F2 ** 2))
    threshold = F1 * ((Ca * np.sqrt(2 * F2 * (h0 ** 2)) / F1) + 1 + (F2 * h0 * (h0 - 1) / F1 ** 2)) ** (1 / h0)
    return threshold


def optimal_components(eigenvalues, percentage):
    sortArray=np.sort(eigenvalues)
    sortArray=sortArray[-1::-1]
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num, sortArray


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
def eval_results(label, residual)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predicted_labels)):
        if (predicted_labels[i] == 1 and labels[i] == 1):
            tp = tp + 1
        if (predicted_labels[i] == 1 and labels[i] == -999):
            fp = fp + 1
        if (predicted_labels[i] == 0 and labels[i] == 1):
            fn = fn + 1
        if (predicted_labels[i] == 0 and labels[i] == -999):
            tn = tn + 1
    return tp, fp, tn, fn

'''
