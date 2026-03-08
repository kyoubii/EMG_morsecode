import sys
print(sys.executable)

from pylsl import StreamInlet, resolve_stream

print("Looking for OpenBCI stream...")

streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])

while True:
    sample, timestamp = inlet.pull_sample()
    print(sample)