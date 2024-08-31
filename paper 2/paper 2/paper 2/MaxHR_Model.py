#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:57:19 2024

@author: mac
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:02:50 2024

@author: mac
"""

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,StackingRegressor
from xgboost import XGBRegressor
from tabulate import tabulate
from sklearn.metrics import mean_absolute_error as mae 

# Load the data from the uploaded files
data_AD8232 = pd.read_csv('Max_HR_AD.csv')
data_Max3003 = pd.read_csv('Max_HR_max.csv')

# Combine the datasets and label them
data_AD8232['sensor'] = 'AD8232'
data_Max3003['sensor'] = 'Max3003'
data = pd.concat([data_AD8232, data_Max3003])

# Prepare the features and target
features = ['Max_HR_sensor']
target = 'Max_HR_emWave'

# Splitting the data for each sensor type
X_AD8232 = data_AD8232[features]
y_AD8232 = data_AD8232[target]

X_Max3003 = data_Max3003[features]
y_Max3003 = data_Max3003[target]

# Splitting the datasets into training and testing sets
X_train_AD8232, X_test_AD8232, y_train_AD8232, y_test_AD8232 = train_test_split(X_AD8232, y_AD8232, test_size=0.2, random_state=42)
X_train_Max3003, X_test_Max3003, y_train_Max3003, y_test_Max3003 = train_test_split(X_Max3003, y_Max3003, test_size=0.2, random_state=42)

# Function to train and evaluate a model
def train_evaluate_model(X_train, X_test, y_train, y_test, model, param_grid, model_name):
    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    # Predicting
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    # Evaluation
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae=mae(y_test, y_test_pred)

    print(f'Model: {model_name}')
    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Training RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')
    print(f'Training R^2: {train_r2}')
    print(f'Test R^2: {test_r2}')
    print(f'Test MAE: {test_mae}')
    print('-'*50)

    return best_model, train_rmse, test_rmse, train_r2, test_r2, test_mae

# Define the models and their hyperparameters
models = {
'LinearRegression': (LinearRegression(), {}),
'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 10, 20], 'min_samples_leaf': [1, 5, 10]}),
'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 10]}),
'SVM': (SVR(), {'C': [0.5, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}),
'kNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
# # 'NeuralNetwork': (MLPRegressor(max_iter=3000, random_state=42), {'hidden_layer_sizes': [(100,), (50, 50)], 'activation': ['relu', 'tanh'], 'alpha': [0.0001, 0.001]}),
'Ridge': (Ridge(), {'alpha': [0.1, 1, 10], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}),
'Lasso': (Lasso(), {'alpha': [0.1, 1, 10]}),
'ElasticNet': (ElasticNet(), {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}),
'GradientBoosting': (GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}),
'AdaBoost': (AdaBoostRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}),
'XGBoost': (XGBRegressor(random_state=42), {'n_estimators': [50,100, 200], 'learning_rate': [0.001,0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]})
}

# Train and evaluate models for AD8232
print("AD8232 Sensor:")
best_models_AD8232 = {}
results_AD8232 = []
for model_name, (model, param_grid) in models.items():
    best_model, train_rmse, test_rmse, train_r2, test_r2, test_mae = train_evaluate_model(X_train_AD8232, X_test_AD8232, y_train_AD8232, y_test_AD8232, model, param_grid, model_name)
    best_models_AD8232[model_name] = best_model
    results_AD8232.append((model_name, train_rmse, test_rmse, train_r2, test_r2,test_mae))

# Train and evaluate models for Max3003
# print(best_models_AD8232)
print("\nMax3003 Sensor:")
best_models_Max3003 = {}
results_Max3003 = []
for model_name, (model, param_grid) in models.items():
    best_model, train_rmse, test_rmse, train_r2, test_r2, test_mae = train_evaluate_model(X_train_Max3003, X_test_Max3003, y_train_Max3003, y_test_Max3003, model, param_grid, model_name)
    best_models_Max3003[model_name] = best_model
    results_Max3003.append((model_name, train_rmse, test_rmse, train_r2, test_r2, test_mae))

# Ensemble model using VotingRegressor
def ensemble_model(X_train, X_test, y_train, y_test, best_models, model_name,chk):
    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ensemble = VotingRegressor(estimators=[(name, model) for name, model in best_models.items()])
    #ensemble = StackingRegressor(estimators=[(name, model) for name, model in best_models.items()])
    ensemble.fit(X_train_scaled, y_train)
     
    y_train_pred = ensemble.predict(X_train_scaled)
    y_test_pred = ensemble.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae=mae(y_test,y_test_pred)

    if chk:
        print(f'Ensemble Model: {model_name}')
        print(f'Training RMSE: {train_rmse}')
        print(f'Test RMSE: {test_rmse}')
        print(f'Training R^2: {train_r2}')
        print(f'Test R^2: {test_r2}')
        print(f'Test R^2: {test_mae}')
        print('-'*50)
    
    return ensemble, train_rmse, test_rmse, train_r2, test_r2, test_mae

# def ensemble_model(X_train, X_test, y_train, y_test, best_models, model_name):
#     try:
#         # Scaling the data
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)

#         # Fit base models and generate meta-features
#         meta_features_train = np.zeros((X_train.shape[0], len(best_models)))
#         meta_features_test = np.zeros((X_test.shape[0], len(best_models)))

#         for i, (name, model) in enumerate(best_models.items()):
#             model.fit(X_train_scaled, y_train)
#             meta_features_train[:, i] = model.predict(X_train_scaled)
#             meta_features_test[:, i] = model.predict(X_test_scaled)

#         # Fit meta-model
#         meta_model = XGBRegressor()
#         meta_model.fit(meta_features_train, y_train)

#         # Predict with meta-model
#         y_train_pred = meta_model.predict(meta_features_train)
#         y_test_pred = meta_model.predict(meta_features_test)

#         # Calculate performance metrics
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         train_r2 = r2_score(y_train, y_train_pred)
#         test_r2 = r2_score(y_test, y_test_pred)

#         # Print the results
#         print(f'Ensemble Model with Meta-Model: {model_name}')
#         print(f'Training RMSE: {train_rmse}')
#         print(f'Test RMSE: {test_rmse}')
#         print(f'Training R^2: {train_r2}')
#         print(f'Test R^2: {test_r2}')
#         print('-'*50)

#         return meta_model, train_rmse, test_rmse, train_r2, test_r2

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     return None, None, None, None, None
    

#dp
store=[]
for model_name, (model, param_grid)  in models.items():
    store.append(model_name)

n=len(store)
minimum_AD8232=1000000

#AD8232 sensor choosing best 3 out of all possible (based on test rmse)
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            model_AD8232={}
            model_AD8232[store[i]]=best_models_AD8232[store[i]]
            model_AD8232[store[j]]=best_models_AD8232[store[j]]
            model_AD8232[store[k]]=best_models_AD8232[store[k]]
            ensemble_AD8232, train_rmse_ensemble_AD8232, test_rmse_AD8232, train_r2_ensemble_AD8232, test_r2_ensemble_AD8232, test_mae_ensemble_AD8232 = ensemble_model(X_train_AD8232, X_test_AD8232, y_train_AD8232, y_test_AD8232, model_AD8232, 'AD8232',0)

            if(test_rmse_AD8232<minimum_AD8232):
                model1_AD8232=store[i]
                model2_AD8232=store[j]
                model3_AD8232=store[k]
                minimum_AD8232 = test_rmse_AD8232


# print(minimum)
# print(model1_AD8232,model2_AD8232,model3_AD8232)

#Max3003 sensor choosing best 3 out of all possible (based on test rmse)
minimum_Max3003=1000000
for i in range(n):
    for j in range(i + 1, n):
        for k in range(j + 1, n):
            model_Max3003={}
            model_Max3003[store[i]]=best_models_Max3003[store[i]]
            model_Max3003[store[j]]=best_models_Max3003[store[j]]
            model_Max3003[store[k]]=best_models_Max3003[store[k]]
            ensemble_Max3003, train_rmse_ensemble_Max3003, test_rmse_Max3003, train_r2_ensemble_Max3003, test_r2_ensemble_Max3003, test_mae_ensemble_Max3003 = ensemble_model(X_train_Max3003, X_test_Max3003, y_train_Max3003, y_test_Max3003, model_Max3003, 'Max3003',0)

            if(test_rmse_Max3003<minimum_Max3003):
                model1_Max3003=store[i]
                model2_Max3003=store[j]
                model3_Max3003=store[k]
                minimum_Max3003 = test_rmse_Max3003

ensemble_model_AD8232={}
ensemble_model_AD8232[model1_AD8232]=best_models_AD8232[model1_AD8232]
ensemble_model_AD8232[model2_AD8232]=best_models_AD8232[model2_AD8232]
ensemble_model_AD8232[model3_AD8232]=best_models_AD8232[model3_AD8232]


ensemble_model_Max3003={}
ensemble_model_Max3003[model1_Max3003]=best_models_Max3003[model1_Max3003]
ensemble_model_Max3003[model2_Max3003]=best_models_Max3003[model2_Max3003]
ensemble_model_Max3003[model3_Max3003]=best_models_Max3003[model3_Max3003]


column_names = ['Model','Training RMSE', 'Test RMSE', 'Training R^2', 'Test R^2', 'Test MAE']

# Create ensemble models
print("\nEnsemble Model for AD8232 Sensor:")
print(model1_AD8232)
print(model2_AD8232)
print(model3_AD8232)
ensemble_AD8232, train_rmse_ensemble_AD8232, test_rmse_ensemble_AD8232, train_r2_ensemble_AD8232, test_r2_ensemble_AD8232, test_mae_ensemble_AD8232 = ensemble_model(X_train_AD8232, X_test_AD8232, y_train_AD8232, y_test_AD8232,ensemble_model_AD8232, 'AD8232',1)

print("\nEnsemble Model for Max3003 Sensor:")
print(model1_Max3003)
print(model2_Max3003)
print(model3_Max3003)
ensemble_Max3003, train_rmse_ensemble_Max3003, test_rmse_ensemble_Max3003, train_r2_ensemble_Max3003, test_r2_ensemble_Max3003, test_mae_ensemble_Max3003 = ensemble_model(X_train_Max3003, X_test_Max3003, y_train_Max3003, y_test_Max3003,ensemble_model_Max3003, 'Max3003',1)

# Summarize results
print("\nSummary of Results for AD8232 Sensor:")
results_AD8232.append(('Ensemble', train_rmse_ensemble_AD8232, test_rmse_ensemble_AD8232, train_r2_ensemble_AD8232, test_r2_ensemble_AD8232,test_mae_ensemble_AD8232))
# for result in results_AD8232:
#     print(f"Model: {result[0]}, Training RMSE: {result[1]}, Test RMSE: {result[2]}, Training R^2: {result[3]}, Test R^2: {result[4]}")

df_AD8232 = pd.DataFrame(results_AD8232, columns=column_names)
print(tabulate(df_AD8232, headers='keys', tablefmt='grid'))


print("\nSummary of Results for Max3003 Sensor:")
results_Max3003.append(('Ensemble', train_rmse_ensemble_Max3003, test_rmse_ensemble_Max3003, train_r2_ensemble_Max3003, test_r2_ensemble_Max3003,test_mae_ensemble_Max3003))
# for result in results_Max3003:
#     print(f"Model: {result[0]}, Training RMSE: {result[1]}, Test RMSE: {result[2]}, Training R^2: {result[3]}, Test R^2: {result[4]}")

df_Max3003 = pd.DataFrame(results_Max3003, columns=column_names)
print(tabulate(df_Max3003, headers='keys', tablefmt='grid'))

