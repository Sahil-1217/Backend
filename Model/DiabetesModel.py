import pickle
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
# from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")


# separating the data and labels

X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']

scaler = StandardScaler()

scaler.fit(X)

std_data = scaler.transform(X)

X = std_data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,stratify=Y ,random_state=2)

model = svm.SVC(kernel='linear')

# training the svm

model.fit(X_train,Y_train)

pickle.dump(model,open('DibetiesModel.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb'))
# model = pickle.load(open('model.pkl','rb'))

