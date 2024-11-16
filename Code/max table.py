#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:10:52 2024

@author: mac
"""

import pandas as pd

# Define the data for the Max3003 Sensor in a structured format
data_max3003 = {
    "Model": ["LinearRegression", "DecisionTree", "RandomForest", "Ridge", "Lasso", "ElasticNet", 
              "GradientBoosting", "AdaBoost", "XGBoost", "Ensemble"],
    "Training RMSE": [29.612875448043724, 0.0, 16.496764221985178, 29.62119141747817, 29.613044293036648, 
                      29.634766568009926, 17.8668313370596, 14.086092963414604, 7.439220750550504, 17.302733784681482],
    "Test RMSE": [38.090120118674406, 21.825447074458747, 25.466460104801403, 38.53803295268537, 38.1537516585298, 
                  38.81845521099891, 28.894928974809485, 33.07891908837947, 22.91138276574336, 27.309664008486294],
    "Training R^2": [0.9276356074178888, 1.0, 0.9775425058221832, 0.9275949585741328, 0.927634782209259, 
                     0.9275285780864322, 0.9736573879248394, 0.9836263741729102, 0.9954331299571924, 0.9752945247392129],
    "Test R^2": [0.28200099005175083, 0.7642642453251054, 0.6790507808301085, 0.2650153848515324, 0.27960007642598683, 
                 0.25428022813611906, 0.5868170200699248, 0.4584958610229821, 0.7402223626261679, 0.6309103431035921]
}

# Create a DataFrame
df_max3003 = pd.DataFrame(data_max3003)

# Format the DataFrame to display all numerical entries up to 3 decimal points
df_max3003_rounded = df_max3003.round(3)

# Save the rounded DataFrame to a CSV file
csv_path_max3003 = 'max3003_results_rounded.csv'
df_max3003_rounded.to_csv(csv_path_max3003, index=False)
