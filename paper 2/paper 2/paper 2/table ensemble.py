#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:39:13 2024

@author: mac
"""

import pandas as pd

# Define the data for the Max3003 Sensor (updated) in a structured format
data_max3003_updated = {
    "Model": ["LinearRegression", "SVM", "Lasso", "XGBoost", "Ensemble"],
    "Training RMSE": [29.612875448043724, 39.521743759732296, 29.613044293036648, 7.439220750550504, 24.574509832701754],
    "Test RMSE": [38.090120118674406, 51.534932847704184, 38.1537516585298, 22.91138276574336, 33.68400911244144],
    "Training R^2": [0.9276356074178888, 0.871105033723544, 0.927634782209259, 0.9954331299571924, 0.9501651116928618],
    "Test R^2": [0.28200099005175083, -0.3143244586661327, 0.27960007642598683, 0.7402223626261679, 0.43850393987001446]
}

# Create a DataFrame
df_max3003_updated = pd.DataFrame(data_max3003_updated)

# Format the DataFrame to display all numerical entries up to 3 decimal points
df_max3003_updated_rounded = df_max3003_updated.round(3)



# Save the transposed DataFrame to a CSV file
csv_path_max3003_updated_transposed = 'ensemble.csv'
df_max3003_updated_rounded.to_csv(csv_path_max3003_updated_transposed, header=False)
