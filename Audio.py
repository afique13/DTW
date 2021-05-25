import matplotlib.pyplot as plt
import numpy as np
import librosa
import IPython.display 
from IPython.display import Image
import librosa.display

y, sr = librosa.load('Audio File\JnTx1.5.wav')
y_harmonic, y_percussive = librosa.effects.hpss(y)

mfcc = librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=13)
print(mfcc)
