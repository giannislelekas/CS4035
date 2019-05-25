import numpy as np
from statsmodels.tsa.arima_model import ARMA

'''
Function to conduct exhaustive grid search for determining the optimal order (p,q)
for the ARMA model to be fitted. The optimal order is found based on AIC; lowest
AIC means bettter fit (for more info on AIC please refer to https://en.wikipedia.org/wiki/Akaike_information_criterion)

INPUT:  signal: time-series (1-D signal) on which ARMA will be fitted,
        p, q: values to be examined
OUTPUT: optimal_p, optimal_q: optimal order extracted,
        AIC: AIC values for each model tested (for evaluation purposes),
        mean_resids, std_resids: mean values and standard deviations for the residuals
                                 from each model
'''
def find_arma_order(signal, p, q):
    AIC = np.zeros((len(p), len(q)))
    mean_resids = np.zeros_like(AIC)
    std_resids = np.zeros_like(AIC)
    for i in range(len(p)):
        for j in range(len(q)):
            model = ARMA(endog=signal, order=(p[i], q[j])).fit(disp=False)
            AIC[i,j] = model.aic
            mean_resids[i,j] = np.mean(model.resid)
            std_resids[i,j] = np.std(model.resid)
    ind = np.unravel_index(np.argmin(AIC), shape=AIC.shape)
    optimal_p = ind[0]
    optimal_q = ind[1]
    return optimal_p, optimal_q, AIC, mean_resids, std_resids


'''
This function performs single step prediction for an ARMA model.
INPUT:  train_signal, test_signal: train and test time-series to fit and predict respectively
        model_p, model_q: order of the ARMA model
        file: provide filepath if you want to store the extracted predictions in an '.npy' file
OUTPUT: predictions: extracted predictions,
        MFE: Mean Forecast error
        MAE: Mean Absolute error
'''
def predict(train_signal, test_signal, model_p, model_q, file=None):

    history = list(train_signal)
    predictions = []
    for t in test_signal:
        model = ARMA(history, order=(model_p, model_q))
        model_fit = model.fit(disp=False)
        pred = model_fit.forecast()[0]

        # Uncomment if you want to see one-step prediction
#         print(f"Actual value: {t}, predicted: {pred}, abs: {np.abs(t-pred)}")
        predictions.append(pred)
        history.append(t)
    predictions = np.array([x for sublist in predictions for x in sublist])

    # Evaluate the predictions
    resid = test_signal - predictions
    MFE = np.mean(resid)
    MAE = np.mean(np.abs(resid/predictions))

    if file != None:
        np.save(file, predictions)
    return predictions, MFE, MAE


'''
This function extracts the threshold based on the maximum residual on the training signal.
INPUT:  train_signal: time-series signal on which to fit the ARMA model,
        model_p, model_q: order of the ARMA model,
        multiplier: value to multiply the maximum residual (to give some room for not extracting FP)
OUTPUT: threshold: extracted threshold
'''
def determine_threshold(train_signal, model_p, model_q, multiplier=1):
    model = ARMA(train_signal, order=(model_p, model_q))
    model_fit = model.fit()

    threshold = multiplier * np.max(np.abs(model_fit.resid))
    return threshold


'''
This function extracts the indices where an alarm has occured, based on the difference
between the test signal and the predicted values, compared to the given threshold.
INPUT:  test_signal, predictions: true and predicted values of the signal
        threshold: set threshold for raising an alarm
OUTPUT: alarm_ind: indices of the test signal where alarm has occured
'''
def extract_alarm_indices(test_signal, predictions, threshold):
    resid = np.abs(test_signal - predictions)
    alarm_ind = resid>threshold
    return alarm_ind
