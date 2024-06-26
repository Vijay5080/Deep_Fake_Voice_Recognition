import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Assuming 'data' is your DataFrame containing features and labels
# ... (your previous code for loading and preprocessing data)

data=pd.read_csv('DATASET-balanced.csv')
# Encode categorical variables
lab = LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i] = lab.fit_transform(data[i])

# Separate features (X) and labels (y)
X = data.drop('LABEL', axis=1)
y = data['LABEL']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SelectKBest with f_classif (ANOVA F-value between label/feature for classification tasks)
x = data.drop('LABEL', axis=1)
y = data['LABEL']
k_best = SelectKBest(f_classif, k=10)
x_train_kbest = k_best.fit_transform(x, y)
selected_features = x.columns[k_best.get_support()]
print(selected_features.values)

# ... (continue with your classifier models)

# Example with Logistic Regression
lr = LogisticRegression(max_iter=500)
lr.fit(x_train_kbest, y_train)
pred = lr.predict(x_test_kbest)
print("The prediction using Logistic regression with SelectKBest: ", pred)
print('The logistic regression with SelectKBest: ', accuracy_score(y_test, pred))
