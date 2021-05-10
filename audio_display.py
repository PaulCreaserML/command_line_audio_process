# Simple script, which receives audio data and displays it

import pyaudio
import wave
import matplotlib.pyplot as plt
import numpy as np
import time
import asyncio
import threading
import queue
from scipy.signal import butter, lfilter


# Audio Parameters
CHUNK = 2048             # 2048 samples
FORMAT = pyaudio.paInt16 # 16 bit
CHANNELS = 1             # Mono audio
RATE = 48000             # 48KHz

# Scipy bufferworth bandpasss filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=True)
    return b, a

# Audio data queue
audio_data_queue = queue.Queue()

# Create filter
filter_a, filter_b = butter_bandpass( 1000, 16000, RATE, order=5)

# Matplotlib
plt.ion() # Stop matplotlib windows from blocking

# Setup figure, axis and initiate plot
fig, ax = plt.subplots()
#xdata, ydata = [], []
#ln, = ax.plot([], [], 'ro-')

counter = 0 # Debug counter

# Callback from pyaudio
def callback(in_data, frame_count, time_info, status):
    global counter, ax, RATE, fig
    tic = time.perf_counter() #  Measure time
    npbuffer = np.frombuffer( in_data, dtype=np.int16)/ (2^16)
    audio_data_queue.put( npbuffer )
    counter += 1 # Not used
    toc = time.perf_counter()
    print(f" {toc - tic:0.4f} seconds") # Print time required to process audio data
    return ( in_data, pyaudio.paContinue )

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback = callback )

print("Start processing")

stream.start_stream()

# wait for stream to finish (5)
while stream.is_active():

    if not audio_data_queue.empty():
        audio_data = audio_data_queue.get()
        audio_data = lfilter( filter_a, filter_b, audio_data )
        fig.clear()
        plt.plot( np.arange(0, len( audio_data)), audio_data  ) # fft_data is a complex number, so the magnitude is computed here
        plt.ylim(-1, 1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    time.sleep(0.01)

print("Finsihed processing")

# Close sterams etc...
stream.stop_stream()
stream.close()
p.terminate()
