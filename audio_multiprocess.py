import multiprocessing

import pyaudio
import wave
import numpy as np
import tensorflow as tf
import time
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
    b, a = butter(order, [low, high], btype='band')
    return b, a



def process1( queue ):

    input = tf.keras.Input(shape=(CHUNK, 1), name="audio")
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(input)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense( 30, activation="relu")(x)
    output = tf.keras.layers.Dense( 10, activation="relu")(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    model.summary()

    while True:
        if not queue.empty():
            tic = time.perf_counter() #  Measure time
            # proc_name = multiprocessing.current_process().name
            audio_data = queue.get()
            audio_data = np.expand_dims( audio_data, axis=0 )
            audio_data = np.reshape( audio_data, (1,CHUNK,1) )
            tensorbuffer = tf.convert_to_tensor( audio_data, dtype=tf.float32)
            model( tensorbuffer )
            toc = time.perf_counter()
            print(f" {toc - tic:0.4f} seconds") # Print time required to process audio data

    time.sleep(0.1)


if __name__ == '__main__':

    # Multi-process queue
    audio_data_queue = multiprocessing.Queue()

    mp = multiprocessing.Process(target=process1, args=(audio_data_queue,))

    # Callback from pyaudio
    def callback(in_data, frame_count, time_info, status):
        global audio_data_queue
        npbuffer = np.frombuffer( in_data, dtype=np.int16)/ (2^16)
        audio_data_queue.put( npbuffer )
        return ( in_data, pyaudio.paContinue )

    mp.start()
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
        time.sleep(0.1)

    # Wait for the worker to finish
    audio_data_queue.close()
    audio_data_queue.join_thread()
    mp.join()
