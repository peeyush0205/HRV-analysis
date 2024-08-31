import pandas as pd

df = pd.read_excel('Untitled spreadsheet.xlsx')

# print(df)
data={
    'RMSSD_emWave': df['RMSSD'][3::2].reset_index(drop=True),
    'RMSSD_sensor': df['RMSSD'][4::2].reset_index(drop=True)
    # 'sensor':[]
}


new_df = pd.DataFrame(data)
# new_df['sensor'] = ''

new_df.to_csv('RMSSD_ad.csv', index=False)
# print(new_df)


#max sensor
df2 = pd.read_excel('Max30003 and emwave pro.xlsx')

# print(df)
data2={
    'RMSSD_emWave': df2['RMSSD'][3::2].reset_index(drop=True),
    'RMSSD_sensor': df2['RMSSD'][4::2].reset_index(drop=True)
    # 'sensor':[]
}


new_df2 = pd.DataFrame(data2)
# new_df['sensor'] = ''

new_df2.to_csv('RMSSD_max.csv', index=False)
# print(new_df)