import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import decomposition




'''
This function detects the anomalies in the signal. It follows the method of the paper 
"Diagnosing Network-Wide Traffic Anomalies" by et al. Lakhina. 
 I: is the identity matrix of the train data 
 P: is a matrix of size m x r where m is the number of principal components and r is the number of normal axes 
 P_T: is the transpose P
 C: is the output of P times P_T
 y_bar: refers to the residual traffic and is (I- C)x y, where y is the test data
 SPE: is the Squared Prediction Error given by squared norm of y_bar
 The anomalies are determined by SPE > threshold.
INPUT: data = normalized train data, data_test = normalized test data,num = number of principal components, threshold
OUTPUT: predicted labels
'''


def det_anomalies(data, data_test, num,threshold):

    model = PCA(n_components= num)
    model.fit(data)
    eigenv = model.components_
    I = np.eye(data.shape[1])
    # P matrix of m x r where r is the number of normal axes and m the number of components
    P = np.transpose(eigenv[0: num-1])
    P_T = np.transpose(P) # P Transpose matrix
    C = np.matmul(P, P_T) # C= P x P_T
    y_bar = np.zeros(data_test.shape)
    SPE = np.zeros(data_test.shape[0])    # squared prediction error
    predicted_labels= np.zeros(data_test.shape[0])
    for i in range(data_test.shape[0]):
        y = data_test[i]
        y_bar[i] = np.matmul(I - C, y)
        SPE[i]= np.square(np.linalg.norm(y_bar[i], 2))
        if SPE[i] > threshold:
            predicted_labels[i]=1

    return predicted_labels


'''
This function computes the spe using a simpler method where y_bar = y - y_hat. 
y_hat: the reconstructed matrix to the original shape of the data based on the number of components 
INPUT: data = normalized data, rec = transformed data back to original space
OUTPUT: residual = Squared Prediction Error
'''


def compute_residuals(data, rec):

    residuals = data - rec
    residuals = np.sum(np.square(residuals),axis=1)
    return residuals


'''
This function implements the Q-statistic function that was developed by Jackson and Mudholkar to produce the threshold
to detect the anomalies. 
L: is the variance captured by projecting the data to the jth principal component
Ca : is a confidence level, we used the same number as in the paper by et al. Lakhina.  
INPUT: pca = PCA model of the original shape of the data, num = number of principal components used.
OUTPUT: Threshold 
'''


def Q_stat(pca, num):

    variance = pca.explained_variance_
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


'''
This function returns the number of principal components that captures the selected percentage of
variance. 
INPUT: eigenvalues = The percentage of variance captured by each principal component of the original shape of the data,
       percentage= the selected percentage.
OUTPUT: num = number of components 
'''


def optimal_components(eigenvalues, percentage):

    sorted_array = np.sort(eigenvalues)
    sorted_array = sorted_array[-1::-1]
    sum_array = sum(sorted_array)
    tmp_sum = 0
    num = 0
    for i in sorted_array:
        tmp_sum += i
        num += 1
        if tmp_sum >= sum_array*percentage:
            return num
