import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import librosa
import numpy as np

def extract_audio_features(audio_file_path):
    y, sr = librosa.load(audio_file_path)
    rms = librosa.feature.rms(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    selected_mfcc_indices = [0, 1, 4, 6, 12, 13, 15, 16, 18, 19]
    selected_mfcc = mfcc[selected_mfcc_indices, :]
    feature_vector = np.concatenate([rms, selected_mfcc.flatten()])
    return feature_vector.tolist()


data = pd.read_csv("DATASET-balanced.csv")
lab = LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i] = lab.fit_transform(data[i])

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

correlation_matrix = data.corr()
sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()'''

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

audio_file_path = 'biden-original.wav'
audio_features = extract_audio_features(audio_file_path)

# Combine RMS and MFCC features into a single feature vector
feature_vector = audio_features['rms'] + np.ravel(audio_features['mfcc']).tolist()
x_train, x_test, y_train, y_test = train_test_split(x, y)

lr = LogisticRegression(max_iter=500)
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
print("The prediction using Logistic regression: ", lr.predict([feature_vector]))
print('The logistic regression: ', accuracy_score(y_test, pred))

lgb = LGBMClassifier()
lgb.fit(x_train, y_train)
prediction = lgb.predict(x_test)
print("The prediction using LGBM: ", lgb.predict([feature_vector]))
print('The LGBM', accuracy_score(y_test, prediction))

tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree.fit(x_train, y_train)
pred = tree.predict(x_test)
print("The prediction using Decision Tree: ", tree.predict([feature_vector]))
print('Decision Tree', accuracy_score(y_test, pred))

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
pred = linear_svc.predict(x_test)
print("The prediction using Linear SVC: ", linear_svc.predict([feature_vector]))
print('The Linear SVC ', accuracy_score(y_test, pred))

mlp_classifier = MLPClassifier()
mlp_classifier.fit(x_train, y_train)
pred = mlp_classifier.predict(x_test)
print("The prediction using MLP classifiers: ", mlp_classifier.predict([feature_vector]))
print('The MLP classifiers ', accuracy_score(y_test, pred))
