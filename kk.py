import librosa
import numpy as np

y, sr = librosa.load('biden-original.wav')
rms = librosa.feature.rms(y=y)[0]
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcs=[]
for i in mfcc:
    sums=sum(i)/len(i)
    mfcs.append(sums)

selected_mfcc_indices = [0, 1, 4, 6, 12, 13, 15, 16, 18, 19]
selected_mfcs =[sum(rms)/len(rms)]+ [mfcs[i] for i in selected_mfcc_indices]
print(selected_mfcs)
'''selected_mfcc = mfcc[selected_mfcc_indices, :]
feature_vector = np.concatenate([rms, selected_mfcc.flatten()])
print(feature_vector)'''