import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display 
from IPython.display import Image

y, sr = librosa.load('ONTIVA.COM_-J_T-beri-penjelasan_-memohon-maaf-320K.wav')
plt.plot(y)
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
IPython.display.Audio(data=y, rate=sr)
