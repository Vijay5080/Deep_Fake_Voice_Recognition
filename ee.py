import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import librosa
import numpy as np

y, sr = librosa.load('biden-original.wav')
rms = librosa.feature.rms(y=y)[0]
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
mfcs = []
for i in mfcc:
    sums = sum(i) / len(i)
    mfcs.append(sums)

selected_mfcc_indices = [0, 1, 4, 6, 12, 13, 15, 16, 18, 19]
selected_mfcs =[sum(rms)/len(rms)]+ [mfcs[i] for i in selected_mfcc_indices]
print(selected_mfcs)

# Rest of the code remains unchanged
data = pd.read_csv("DATASET-balanced.csv")
lab = LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i] = lab.fit_transform(data[i])

noise = {}
for i in data.select_dtypes(include='number').columns.values:
    data['z-scores'] = (data[i] - data[i].mean()) / (data[i].std())
    out = np.abs(data['z-scores'] > 3)
    if out.sum() > 0:
        noise[i] = out.sum()

thresh = 2.5
for i in noise:
    upper = data[i].mean() + thresh * data[i].std()
    lower = data[i].mean() - thresh * data[i].std()
    data = data[(data[i] > lower) & (data[i] < upper)]

lab = LabelEncoder()
for i in data.select_dtypes(include="object").columns.values:
    data[i] = lab.fit_transform(data[i])

x = []
corr = data.corr()['LABEL']
corr = corr.drop(['LABEL', 'z-scores'])
for i in corr.index:
    if corr[i] > 0:
        x.append(i)

print(x)
x = data[x]
y = data['LABEL']

x_train, x_test, y_train, y_test = train_test_split(x, y)

lr = LogisticRegression(max_iter=500)
lr.fit(x_train, y_train)
pred = lr.predict([selected_mfcs])
print("The prediction using Logistic regression: ", pred)
print('The logistic regression: ', accuracy_score(y_test, lr.predict(x_test)))
