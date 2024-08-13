#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:24:13 2024

@author: mac
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import pymc as pm
from sklearn.metrics import mean_squared_error

# Example data - replace with your actual data
data = pd.read_csv('em_AD.csv')
sensor1 = data.iloc[:, 0]
sensorX = data.iloc[:, 1]
data = pd.DataFrame({'sensor1': sensor1, 'sensorX': sensorX})
# Copy the first column into a NumPy array
sensor1_array = data.iloc[:, 0].to_numpy()

def bayesian_regression(X_train, y_train):
    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Expected value of outcome
        mu = alpha + beta * X_train

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y_train)

        # Posterior distribution
        trace = pm.sample(1000, return_inferencedata=False)
    
    return model, trace

kf = KFold(n_splits=3, shuffle=True, random_state=42)
mse_scores = []

for train_index, test_index in kf.split(data):
    X_train, X_test = data['sensor1'].values[train_index], data['sensor1'].values[test_index]
    y_train, y_test = data['sensorX'].values[train_index], data['sensorX'].values[test_index]
    
    # Fit the Bayesian regression model
    model, trace = bayesian_regression(X_train, y_train)
    
    # Make predictions
    with model:
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['Y_obs'])
        y_pred = np.mean(posterior_predictive['Y_obs'], axis=0)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Print average MSE
print(f'Average MSE: {np.mean(mse_scores)}')