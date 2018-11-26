import numpy as np
import pyaudio
import matplotlib.pyplot as plt

def makeWave(freq=440, duration=1, samplingRate=44100):
	data = np.zeros(duration*samplingRate)
	for i in range(len(data)):
		data[i] += np.sin(2*(np.pi)*freq*i/samplingRate)

	# plt.plot(data[0:1000])
	# plt.show()

	# T = duration
	# fs = samplingRate
	# t = np.arange(0,T,1/fs)
	# x = 0.25 * np.sin(2*np.pi*440*t)

	return data
	

PyAudio = pyaudio.PyAudio

samplingRate = 44100
duration = 1

data = makeWave(duration=3, freq=220)
# data += makeWave(duration=3, freq=660)
# data += makeWave(duration=3, freq=880)
# data += makeWave(duration=3, freq=1100)
# data += makeWave(duration=3, freq=1320)
# data += 10*makeWave(duration=3, freq=1540)

data /= np.amax(data)

p = PyAudio()
stream = p.open(format = pyaudio.paFloat32, 
                channels = 1, 
                rate = samplingRate, 
                output = True)

data = data.astype(np.float32).tobytes()# 	(data*2147483648).astype(np.float32).tobytes()
stream.write(data)
stream.stop_stream()
stream.close()
p.terminate()
