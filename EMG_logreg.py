import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression

file_path = r"C:\Users\foxki\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2026-03-07_16-13-26\BrainFlow-RAW_2026-03-07_16-13-26_6.csv"

# Load raw lines
raw = pd.read_csv(file_path, skiprows=1, header=None)

# Remove quotes and split tab-separated values
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

left_filtered = filtfilt(b, a, left_raw)
right_filtered = filtfilt(b, a, right_raw)

left_rect = np.abs(left_filtered)
right_rect = np.abs(right_filtered)

left_env = pd.Series(left_rect).rolling(200).mean().dropna()
right_env = pd.Series(right_rect).rolling(200).mean().dropna()

offset = len(left_rect) - len(left_env)

left_rect = left_rect[offset:]
right_rect = right_rect[offset:]

window = 200  # samples (~200 ms at 1000 Hz)

features = []
targets = []

combined_env = (left_env + right_env) / 2
baseline = np.median(combined_env)
noise = np.std(combined_env)

threshold = baseline + 0.6 * noise

max_len = min(len(left_rect), len(right_rect), len(left_env), len(right_env))
signal = combined_env.values

start_thresh = threshold
stop_thresh = threshold * 0.7

labels = np.zeros(len(signal))
active = False
start = 0

for i, v in enumerate(signal):

    if not active and v > start_thresh:
        active = True
        start = i

    elif active and v < stop_thresh:
        duration = i - start

        if duration < 120:
            labels[start:i] = 1
        else:
            labels[start:i] = 2

        active = False

for i in range(window, max_len):
    seg_right = right_rect[i-window:i]
    seg_left = left_rect[i-window:i]

    rms_left = np.sqrt(np.mean(seg_left**2))
    mean_left = np.mean(seg_left)
    peak_left = np.max(seg_left)
    std_left = np.std(seg_left)

    rms_right = np.sqrt(np.mean(seg_right**2))
    mean_right = np.mean(seg_right)
    peak_right = np.max(seg_right)
    std_right = np.std(seg_right)

    features.append([rms_left, mean_left, peak_left, std_left,
                     rms_right, mean_right, peak_right, std_right])

    # no signal
    targets.append(labels[i])
if start is not None:
    duration = len(signal) - start
    if duration < 120:
        labels[start:] = 1
    else:
        labels[start:] = 2


features = np.array(features)
targets = np.array(targets)

features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

model = LogisticRegression(max_iter = 1000)
model.fit(features, targets)

predictions = model.predict(features)

plt.figure(figsize=(12,6))

plt.plot(combined_env, label="Muscle Activity")

# scale labels so they appear clearly on the graph
label_scale = np.max(combined_env) / 4
plt.step(range(len(labels)), labels * label_scale, where="post", label="Detected Morse")

plt.title("Automatic EMG Morse Code Detection")

# Morse classification ticks
plt.yticks(
    [0, label_scale, 2*label_scale],
    ["Pause", "Dot", "Dash"]
)

plt.xlabel("Samples")
plt.ylabel("EMG Activity")

plt.legend()
plt.show()