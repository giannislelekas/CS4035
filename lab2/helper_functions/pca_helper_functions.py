import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import decomposition




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


def compute_residuals(data, rec):

    residuals = data - rec
    residuals = np.sum(np.square(residuals),axis=1)
    return residuals



'''
This function ....
INPUT: ....
OUTPUT
'''
def Q_stat(pca, num):

    variance = pca.explained_variance_
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
