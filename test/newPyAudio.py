import pyaudio
import numpy as np
fs = 44100 # Hz
T = 1. # second, arbitrary length of tone

# 1 kHz sine wave, 1 second long, sampled at 8 kHz
t = np.arange(0,T,1/fs)
x = 0.25 * np.sin(2*np.pi*1100*t)   # 0.5 is arbitrary to avoid clipping sound card DAC
x += 0.25 * np.sin(2*np.pi*440*t)
x += 0.25 * np.sin(2*np.pi*660*t)
x  = (x*32768).astype(np.int16)  # scale to int16 for sound card



# PyAudio doesn't seem to have context manager
P = pyaudio.PyAudio()

stream = P.open(rate=fs, format=pyaudio.paInt16, channels=1, output=True)
stream.write(x.tobytes())

stream.close() # this blocks until sound finishes playing

P.terminate()