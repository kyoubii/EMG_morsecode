import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

file_path = r"C:\Users\foxki\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2026-03-07_16-13-26\BrainFlow-RAW_2026-03-07_16-13-26_6.csv"

# Read raw lines
raw = pd.read_csv(file_path, skiprows=1, header=None)

# Remove quotes/take in string as float number
data = raw[0].str.replace('"', '').str.split('\t', expand=True)
data = data.astype(float)

# Extract channels
time = data[13]
left_raw = data[1]
right_raw = data[4]

# Bandpass filter (EMG standard)
fs = 1000
low = 20
high = 450
b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')

# Assign filtered values to variables
left_filtered = filtfilt(b, a, left_raw)
right_filtered = filtfilt(b, a, right_raw)

# Plot
plt.figure(figsize=(12,8))

plt.subplot(2,1,1)
plt.title("Raw EMG")
plt.plot(time, left_raw, label="Left Hand")
plt.plot(time, right_raw, label="Right Hand")
plt.legend()

plt.subplot(2,1,2)
plt.title("Filtered EMG")
plt.plot(time, left_filtered, label="Left Hand")
plt.plot(time, right_filtered, label="Right Hand")
plt.legend()

plt.xlabel("Time")
plt.tight_layout()

plt.show()
