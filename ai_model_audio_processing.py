# Siumple script, which receives audio data and processes it using an AI model_build
# The model is not trained, so it does not produce useful output

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

# Functional AI model
def model_build():
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
    return model

# Tensorflow Lite
def  tflite_conversion( model ):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    return interpreter, input_index, output_index

# Build Model
model = model_build()
# Display Model
model.summary()
# Create tflite model
interpreter, input_index, output_index = tflite_conversion( model )
# Create filter
filter_a, filter_b = butter_bandpass( 1000, 16000, RATE, order=5)

counter = 0 # Debug counter

# Callback from pyaudio
def callback(in_data, frame_count, time_info, status):
    global counter, input_index, output_index, interpreter
    tic = time.perf_counter() #  Measure time
    npbuffer = np.frombuffer( in_data, dtype=np.int16)/ (2^16)
    npbuffer = lfilter( filter_b, filter_a, npbuffer)
    npbuffer = np.expand_dims( npbuffer, axis=0 )
    npbuffer = np.reshape( npbuffer, (1,CHUNK,1) )  # Add necessary dimensions for batch, 1D to 2D audio conversion
    tensorbuffer = tf.convert_to_tensor( npbuffer, dtype=tf.float32)
    interpreter.set_tensor(input_index, tensorbuffer)
    interpreter.invoke()
    interpreter.get_tensor(output_index) # Output
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
    time.sleep(0.1)

print("Finsihed processing")

# Close sterams etc...
stream.stop_stream()
stream.close()
p.terminate()
