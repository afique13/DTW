import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display 
from IPython.display import Image
import librosa.display
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from DTW import dtw

y1, sr1 = librosa.load('Audio File\JnTExpressx1.wav')
y2, sr2 = librosa.load('Audio File\JnTExpressx0.5.wav')
y3, sr3 = librosa.load('Audio File\MemohonMaafx1.wav')

y_harmonic1, y_percussive1 = librosa.effects.hpss(y1)
y_harmonic2, y_percussive2 = librosa.effects.hpss(y2)
y_harmonic3, y_percussive3 = librosa.effects.hpss(y3)

mfcc1 = librosa.feature.mfcc(y=y_percussive1, sr=sr1, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=y_percussive2, sr=sr2, n_mfcc=13)
mfcc3 = librosa.feature.mfcc(y=y_percussive3, sr=sr3, n_mfcc=13)

x = np.amax(mfcc1, axis=1)
y = np.amax(mfcc2, axis=1)
z = np.amax(mfcc3, axis=1)

# distance1, path1 = fastdtw(x, y, dist=None)
# distance2, path2 = fastdtw(x, z, dist=None)

# print(distance1)
# print(distance2)

# print(dtw(x,y))
# print()
# print(dtw(x,z))

x_matrix = np.array(dtw(x,y))
y_matrix = np.array(dtw(x,z))

print(x_matrix[x_matrix.shape[0]-1,x_matrix.shape[1]-1])
print(y_matrix[y_matrix.shape[0]-1,y_matrix.shape[1]-1])