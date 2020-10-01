# python script to run my evaluation code for regression functions

# Libraries needed:
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pydataset import data


import math
from sklearn.metrics import mean_squared_error
from math import sqrt


from statsmodels.formula.api import ols

# Function is broken into two parts; first is the inital prep stage, such as setting the baseline model, then the functions roll through determining
# if the baseline model is better or the super simple model is the best, and after that the function will run through evaluating the manufactured model to see if it's better 
# than the baseline model.

def model_function(linear_model, target, feature):
    
    '''
    This function will take in a linear model, a target variable, and a feature. It will calculate if the baseline model or the super simple model is a better fit for the data. If the user's model is a better fit, will run through the SSE, MSE, and RMSE calucations to generate model metrics and check model significance, and return a scatterplot of the model on the dataset.    
    '''
    
    baseline = target.mean()
    model = linear_model
    evaluate = pd.DataFrame()
    evaluate['x'] = feature
    
    # y variable is sales, and the baseline is the mean of sales.
    evaluate["y"] = target
    evaluate["baseline"] = target.mean()

    # y-hat is a common shorthand for "predicted y" values in statistics
    evaluate['yhat'] = model.predict()

    # Calc the baseline residuals (errors)
    evaluate["baseline_residual"] = evaluate.baseline - evaluate.y

    # Calc the model's residuals:
    evaluate["model_residual"] = evaluate.yhat - evaluate.y
    
    # Now the function calculates the SSE, MSE, and RMSE:
    # Calculate if the model beats the baseline
    # Square errors for 2 reasons:
        # 1. Squaring large numbers increases their magnitude (opposite for small)
        # 2. Squaring removes negative residuals
        
    baseline_sse = (evaluate.baseline_residual**2).sum()
    model_sse = (evaluate.model_residual**2).sum()

    if model_sse > baseline_sse:
        print("Our baseline is better than the model")
    
    else:
        print("Our model beats the baseline")
        
        metrics = pd.DataFrame()
        
        # Calculating SSE
        model_sse = (evaluate.model_residual**2).sum()

        # Calcuating MSE
        mse = mean_squared_error(evaluate.y, evaluate.yhat)

        # Calculating RMSE
        rmse = sqrt(mse)

        print("SSE is", model_sse, " which is the sum sf squared errors")
        print("MSE is", mse, " which is the average squared error")
        print("RMSE is", rmse, " which is the square root of the MSE")
        
        # Evaluate the model significance using r-squared.
        r2 = model.rsquared
        print('R-squared =', round(r2, 3))
        
        
        # Now to evaluate the p-value:
        f_pval = model.f_pvalue
        print("p-value for model significance = ", f_pval)
        
        
        # Nested function to generate plot:
        def plot_residuals(actual, predicted):
            residuals = actual - predicted
            plt.hlines(0, actual.min(), actual.max(), ls = ':')
            plt.scatter(actual, residuals)
            plt.ylabel('residual ($ - \hat{y}$)')
            plt.xlabel('actual value ($y$)')
            plt.title('Actual vs Residual')
            return plt.gca()
        
        actual = evaluate.y
        predicted = evaluate.yhat
        plot_residuals(evaluate.y, evaluate.yhat)

# these functions require that an evaluation dataframe is already setup.
# For plotting the residuals
def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    return plt.gca()

# these functions require that an evaluation dataframe is already setup.
def regression_errors(y, yhat):
    
    # Sum of squared errors (SSE):
    model_sse = (df.model_residual**2).sum()
    
    # mean squared error (MSE)
    model_mse = model_see/len(df)

    # root mean squared error (RMSE)
    model_rmse = sqrt(model_mse)
    
    # explained sum of squares (ESS)
    model_ess = sum((df.yhat - df.y.mean())**2)
    
    # total sum of squares (TSS)
    model_tss = model_ess + model_see


# these functions require that an evaluation dataframe is already setup.
def baseline_mean_errors(y):
    # Sum of squared errors (SSE):
    baseline_sse = (df.baseline_residual**2).sum()
    
    # mean squared error (MSE)
    baseline_mse = baseline_see/len(df)

    # root mean squared error (RMSE)
    baseline_rmse = sqrt(baseline_mse)
    
    return baseline_see, baseline_mse, baseline_rmse


# these functions require that an evaluation dataframe is already setup.
def better_than_baseline(y, yhat, df):
    
    # Model metrics:
    
    # Sum of squared errors (SSE):
    model_sse = (df.model_residual**2).sum()
    
    # mean squared error (MSE)
    model_mse = model_sse/len(df)

    # root mean squared error (RMSE)
    model_rmse = sqrt(model_mse)
    
    # explained sum of squares (ESS)
    model_ess = sum((df.yhat - df.y.mean())**2)
    
    # total sum of squares (TSS)
    model_tss = model_ess + model_sse
    

    # Baseline metrics:
    # Sum of squared errors (SSE):
    baseline_sse = (df.baseline_residual**2).sum()
    
    # mean squared error (MSE)
    baseline_mse = baseline_sse/len(df)

    # root mean squared error (RMSE)
    baseline_rmse = sqrt(baseline_mse)
    
    if model_sse < baseline_sse:
        print("Our model beats the baseline.")
    else:
        print("Our baseline is better than the model.")
    print("Baseline SSE", baseline_sse)
    print("Model SSE", model_sse)