# Import necessary libraries
import neurokit2 as nk
import matplotlib.pyplot as plt

# Generate a synthetic ECG signal
ecg_signal = nk.ecg_simulate(duration=30, sampling_rate=1000)  # Increase duration to 30 seconds

# Process the ECG signal to extract heartbeats
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=1000)
heartbeats = nk.ecg_findpeaks(ecg_cleaned, sampling_rate=1000)

# Ensure there are enough peaks to calculate HRV
if len(heartbeats["ECG_R_Peaks"]) < 2:
    raise ValueError("Not enough R-peaks detected to calculate HRV.")

# Compute heart rate variability (HRV) metrics
hrv = nk.hrv(heartbeats["ECG_R_Peaks"], sampling_rate=1000)

# Display HRV metrics
# print("HRV Metrics:")
# print(hrv)

hrv_dict = hrv.to_dict()
print(hrv_dict)

# Alternatively, convert to a list of tuples
hrv_list = list(hrv.itertuples(index=False, name=None))
print(hrv_list)

# Plot the original ECG signal and the detected heartbeats
plt.figure(figsize=(12, 6))

# Plot ECG Signal
plt.subplot(2, 1, 1)
plt.plot(ecg_signal, label='Synthetic ECG Signal')
plt.plot(heartbeats["ECG_R_Peaks"], ecg_cleaned[heartbeats["ECG_R_Peaks"]], "ro", label='Detected Peaks')
plt.title('Synthetic ECG Signal with Detected Peaks')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend()

# Plot HRV metrics
plt.subplot(2, 1, 2)
hrv_metrics = hrv[['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD']]  # Extract specific metrics
hrv_metrics.plot(kind='bar')
plt.title('Heart Rate Variability (HRV) Metrics')
plt.ylabel('Milliseconds')
plt.xticks(rotation=45)
plt.legend(title='HRV Metrics')

plt.tight_layout()
plt.show()
