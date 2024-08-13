#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:23:05 2024

@author: mac
"""

import pandas as pd

# Define the updated data for the Max3003 Sensor in a structured format
data_max3003_final = {
    "Model": ["LinearRegression", "SVM", "Lasso", "Ensemble"],
    "Training RMSE": [29.612875448043724, 39.521743759732296, 29.613044293036648, 30.880590517884407],
    "Test RMSE": [38.090120118674406, 51.534932847704184, 38.1537516585298, 42.376832394733775],
    "Training R^2": [0.9276356074178888, 0.871105033723544, 0.927634782209259, 0.92130720799681],
    "Test R^2": [0.28200099005175083, -0.3143244586661327, 0.27960007642598683, 0.1112980310202697]
}

# Create a DataFrame
df_max3003_final = pd.DataFrame(data_max3003_final)

# Format the DataFrame to display all numerical entries up to 3 decimal points
df_max3003_final_rounded = df_max3003_final.round(3)

# Save the rounded DataFrame to a CSV file
csv_path_max3003_final = 'AD ensemble.csv'
df_max3003_final_rounded.to_csv(csv_path_max3003_final, index=False)
