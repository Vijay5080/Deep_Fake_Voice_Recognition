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
from xgboost import XGBClassifier
from keras.models import Sequential
from lazypredict.Supervised import  LazyClassifier
from keras.layers import Dense
import keras.activations,keras.losses,keras.optimizers
import librosa
from keras.models import load_model
from sklearn.feature_selection import SelectKBest, f_classif

y, sr = librosa.load('linus-original-DEMO.mp3')
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
selected_mfccs = [1, 3, 5, 9, 11, 17]
mfcs=[]

for i in mfcc:
    mfcs.append(np.median(i))
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

features =[np.var(centroid)]+[np.var(bandwidth)]+[np.var(roll_off)]+[np.var(cross)]+ [mfcs[i] for i in selected_mfccs]
print(features)

data=pd.read_csv("DATASET-balanced.csv")
print(data.columns)
print(data.describe())
print(data.isna().sum())
print(data.describe())
print(data.info())
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

noise={}
for i in data.select_dtypes(include='number').columns.values:
    data['z-scores']=(data[i]-data[i].mean())/(data[i].std())
    out=np.abs(data['z-scores'] > 3)
    if out.sum() >0:
        noise[i]=out.sum()

thresh=3
for i in noise:
    upper=data[i].mean()+thresh*data[i].std()
    lower=data[i].mean()-thresh*data[i].std()
    data=data[(data[i]>lower)&(data[i]<upper)]
lab=LabelEncoder()

for i in data.select_dtypes(include="object").columns.values:
    data[i]=lab.fit_transform(data[i])

x = data.drop(['LABEL','z-scores'], axis=1)
y = data['LABEL']

k_best = SelectKBest(f_classif, k=10)
x_train_kbest = k_best.fit_transform(x, y)
selected_features = x.columns[k_best.get_support()]
print(selected_features.values)

x=data[selected_features]
y=data['LABEL']
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)
lr = LogisticRegression(max_iter=500)
lr.fit(x_train, y_train)
pred=lr.predict([features])
print("The prediction using Logistic regression: ",pred)
print('The logistic regression: ', lr.score(x_test, y_test))

lgb = LGBMClassifier()
lgb.fit(x_train, y_train)
prediction = lgb.predict([features])
print("The prediction using Lgb: ",prediction)
print('The LGB', lgb.score(x_test, y_test))

tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree.fit(x_train, y_train)
pred=tree.predict([features])
print("The prediction using dtree: ",pred)
print('Dtree ', tree.score(x_test,y_test))

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
pred=linear_svc.predict([features])
print("The prediction using Linear_svc: ",pred)
print('The Linear SVC ',linear_svc.score(x_test,y_test))

mlp_classifier = MLPClassifier()
mlp_classifier.fit(x_train, y_train)
pred=mlp_classifier.predict([features])
print("The prediction using MLP: ",pred)
print('The MLP classifiers ',mlp_classifier.score(x_test,y_test))


lazy=LazyClassifier()
models,predict=lazy.fit(x_train,x_test,y_train,y_test)
print(models)

Y=pd.get_dummies(y)
x_tran,x_tst,y_tran,y_tst=train_test_split(x,Y)
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.tanh))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=Y.shape[1],activation=keras.activations.sigmoid))
models.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics='accuracy')
hist=models.fit(x_tran,y_tran,batch_size=20,epochs=40)
models.save("DEEP_Fake.h5")