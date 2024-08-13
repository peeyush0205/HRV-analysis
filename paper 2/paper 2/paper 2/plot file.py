#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 01:26:23 2024

@author: mac
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'em_AD.csv'  # replace with your file path
data = pd.read_csv(file_path)

# Check if the file has exactly 2 columns
if data.shape[1] != 2:
    raise ValueError("CSV file must have exactly two columns")

# Extract the columns
col1 = data.iloc[:, 0]
col2 = data.iloc[:, 1]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(col2, col1, label='Data Points', color='blue')
plt.title('Mean RR estimation with emWave Pro and AD8232')
plt.xlabel('Column 2')
plt.ylabel('Column 1')
plt.legend()
plt.grid(True)
plt.show()