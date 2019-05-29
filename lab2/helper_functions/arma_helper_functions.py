import numpy as np
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
            try:
                model = ARMA(endog=signal, order=(p[i], q[j])).fit(disp=False)
                AIC[i,j] = model.aic
                mean_resids[i,j] = np.mean(model.resid)
                std_resids[i,j] = np.std(model.resid)
            except:
                print(f"Failed to fit ARMA model {(i, j)}")
    ind = np.argmin(AIC)
    ind_p, ind_q = np.unravel_index(ind, AIC.shape)
    optimal_p, optimal_q = p[ind_p], q[ind_q]

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
def predict(train_signal, test_signal, model_p, model_q, file=None, print_output=False):

    history = list(train_signal)
    predictions = []
    for t in test_signal:
        model = ARMA(history, order=(model_p, model_q))
        model_fit = model.fit(disp=False)
        pred = model_fit.forecast()[0]

        if print_output:
            print(f"Actual value: {t}, predicted: {pred}, abs: {np.abs(t-pred)}")
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



def predictS(train_signal, test_signal, model_p, model_q, file=None, print_output=False):

    history = list(train_signal)
    predictions = []

    model_1 = SARIMAX(history, order=(model_p, 0, model_q), enforce_stationarity=False, enforce_invertibility=False)
    res = model_1.fit()
    i = 0
    for t in test_signal:
        # model_2 = SARIMAX(history, order=(model_p, 0, model_q))
        model_2 = SARIMAX(test_signal[:i], order=(model_p, 0, model_q), enforce_stationarity=False, enforce_invertibility=False)
        res2 = model_2.filter(res.params)
        pred = res2.forecast(1)

        if print_output:
            print(f"Actual value: {t}, predicted: {pred}, abs: {np.abs(t-pred)}")
        predictions.append(pred)
        history.append(t)
        i = i+1
    predictions = np.array([x for sublist in predictions for x in sublist])

    # Evaluate the predictions
    resid = test_signal - predictions
    MFE = np.mean(resid)
    MAE = np.mean(np.abs(resid))
    MAPE = np.round(np.mean(np.abs(resid/(test_signal + 1e-16))), 5 )

    if file != None:
        np.save(file, predictions)
    return predictions, MFE, MAE, MAPE



def predict_mini(train_signal, test_signal, model_p, model_q, file=None, print_output=False):

    history = list(train_signal)
    model = ARMA(history, order=(model_p, model_q)).fit()
    ar_coef = model.arparams
    ma_coef = model.maparams

    predictions = []
    residuals = list(model.resid)


    for t in test_signal:

        # mu = np.mean(history)

        if len(ar_coef) != 0 and len(ma_coef)!=0:
            pred = sum(ar_coef * history[-model_p:]) + sum(ma_coef * residuals[-model_q:])
        elif len(ar_coef) == 0 and len(ma_coef)!=0:
            pred = sum(ma_coef * residuals[-model_q:])
        elif len(ar_coef) != 0 and len(ma_coef)==0:
            pred = sum(ar_coef * history[-model_p:])
        else:
            print("ARMA (0,0) given")
            return

        if print_output:
            print(f"Actual value: {t}, predicted: {pred}, abs: {np.abs(t-pred)}")
        predictions.append(pred)
        history.append(t)
        residuals.append(t-pred)
    # predictions = [x for sublist in predictions for x in sublist]
    predictions = np.array(predictions)

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

    mean_res = np.mean(model_fit.resid)
    std_res = np.std(model_fit.resid)
    threshold_up = mean_res + multiplier*std_res
    threshold_down = mean_res - multiplier*std_res

    return threshold_up, threshold_down


'''
This function extracts the indices where an alarm has occured, based on the difference
between the test signal and the predicted values, compared to the given threshold.
INPUT:  test_signal, predictions: true and predicted values of the signal
        threshold: set threshold for raising an alarm
OUTPUT: alarm_ind: indices of the test signal where alarm has occured
'''
def extract_alarm_indices(test_signal, predictions, threshold_up, threshold_down):
    resid = test_signal - predictions
    alarm_ind = np.logical_or(resid > threshold_up, resid<threshold_down)

    return alarm_ind
