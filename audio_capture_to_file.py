import pyaudio
import wave

# Blocking audio recorder

# Audio Parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording Started")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    # Saves all data into buffer, alternatively the data can be written to the audio filename
    # It important not to spend too much time processing between reads
    # If reads are insufficient, a buffer flow will occur ( exception)
    data = stream.read(CHUNK)
    frames.append(data)

print("Recoerding complete")

stream.stop_stream()
stream.close()
p.terminate()

# Save buffered data to audio file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
