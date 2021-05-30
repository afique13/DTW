import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display 
from IPython.display import Image
import librosa.display
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

y1, sr1 = librosa.load('Audio File\JnTx1.wav')
y2, sr2 = librosa.load('Audio File\JnTx2.wav')
y_harmonic1, y_percussive1 = librosa.effects.hpss(y1)
y_harmonic2, y_percussive2 = librosa.effects.hpss(y2)

mfcc1 = librosa.feature.mfcc(y=y_percussive1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y_percussive2, sr=sr2, n_mfcc=13)

#print(mfcc1)
#print(mfcc2)

# plt.figure(figsize=(14,5))

# plt.plot(y_percussive1)
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.show()

# plt.figure(figsize=(14,5))
# plt.plot(y_percussive2)
# plt.xlabel('Time (samples)')
# plt.ylabel('Amplitude')
# plt.show()

x = np.amax(mfcc1, axis=1)
y = np.amax(mfcc2, axis=1)
distance, path = fastdtw(x, y, dist=euclidean)
print(distance)
