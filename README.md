# command_line_audio_process
Simple python scripts demonstrate receiving and processing of audio data

## Modules

Makes use of pyaudio

On windows this can be a little awkward to install, so pipwin is recommended for installation

pip install pipwin
pipwin install pyaudio

## Scripts

There are a number of demonstration scripts. The aim is not to provide solutions, but reference code, from which applications can be built.

Ideally the aim to produce an AI system, which processes audio in real time on edge devices.

## Example Programs

These are examples are not finished applications. However they provide a template for which trained models can be used to produce simple demo applications.

### Audio Save

This simply takes audio and saves it to an audio file

### Audio AI Model Processing

Simply takes audio data, filters using a bandpass filter and processes it using an untrained model.

### Audio AI Model Processing & GUI

Simply takes audio data, filters using a bandpass filter and processes it using an untrained model.

It then provides a possible option displaying the results in a GUI ( Uses TKinter )
