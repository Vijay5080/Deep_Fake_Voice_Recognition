import librosa
import numpy as np
y, sr = librosa.load('Obama-to-Trump.wav')
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
selected_mfccs = mfcc[[1, 3, 5, 9, 11, 17]]
selected_mfcc_indices = [0, 1, 4, 6, 12, 13, 15, 16, 18, 19]
mfcs=[]
for i in mfcc:
    mfcs.append(np.mean(i))
centroid=[]
bandwidth=[]
cross=[]
for i in spectral_centroid:
    centroid.append(np.mean(spectral_centroid))
for i in spectral_bandwidth:
    bandwidth.append(np.mean(i))
for i in zero_crossing_rate:
    cross.append(np.mean(i))
roll_off=[]
for i in rolloff:
    roll_off.append(np.mean(i))

features =[np.mean(centroid)]+[np.mean(bandwidth)]+[np.mean(roll_off)]+[np.mean(cross)]+ [mfcs[i] for i in selected_mfcc_indices]
print(features)