import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path="linus-original-DEMO.mp3"
y, sr = librosa.load(audio_path)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title(f'Waveform of  {audio_path}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

audio_path="linus-to-musk-DEMO.mp3"
y, sr = librosa.load(audio_path)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title(f'Waveform of  {audio_path}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()


plt.figure(figsize=(10, 4))
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()