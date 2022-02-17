# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:46:38 2022

@author: postulate-41
"""

import os
import pandas as pd 
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout, Flatten,TimeDistributed, MaxPooling1D,Conv1D, MaxPool2D, BatchNormalization,MaxPooling2D,Bidirectional


data=pd.read_csv('Train.csv',index_col='Reading_ID')

X=data.iloc[:,:-3]
Y=data.iloc[:,-3:]

# y=pd.get_dummies(Y)


col=Y.columns

unique=list(Y[col[2]].unique())
Y=Y.replace(unique,[1,2,3])

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.10,shuffle=True)
# Model = keras.Sequential([
#     keras.layers.Dense(32,input_dim=X.shape[1], activation='relu'),
#     keras.layers.Dense(units=64, activation='relu'),
#     keras.layers.Dense(units=128, activation='relu'),
#     keras.layers.Dense(units=256, activation='relu'),
#     keras.layers.Dense(units=512, activation='relu'),
#     keras.layers.Dense(units=1024, activation='relu'),
#     keras.layers.Dense(y.shape[1], activation='softmax')
# ])

# Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# print(Model.summary())


# history = Model.fit(xtrain, ytrain, batch_size=128, epochs=20, validation_data=(xtest,ytest))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=20)
clf.fit(xtrain,ytrain)
pred=clf.predict(xtrain)

dt=pd.DataFrame(pred,columns=col)

from sklearn.metrics import classification_report, confusion_matrix
print("===================== hdl_cholesterol_human =================================")
print(confusion_matrix(pred[:,0],ytrain.iloc[:,0]))
print(classification_report(pred[:,0],ytrain.iloc[:,0]))

print("===================== hemoglobin(hgb)_human =================================")
print(confusion_matrix(pred[:,1],ytrain.iloc[:,1]))
print(classification_report(pred[:,1],ytrain.iloc[:,1]))

print("===================== cholesterol_ldl_human =================================")
print(confusion_matrix(pred[:,2],ytrain.iloc[:,2]))
print(classification_report(pred[:,2],ytrain.iloc[:,2]))

dt2=pd.read_csv('Test.csv',index_col='Reading_ID')
pred=clf.predict(dt2)
Reading_ID=list(dt2.index.values)

Reading_ID_hdl_cholesterol_human=[]
Reading_ID_hemoglobin_hgb__human=[]
Reading_ID_cholesterol_ldl_human=[]
for i in Reading_ID:
    Reading_ID_hdl_cholesterol_human.append(i+"_hdl_cholesterol_human")
    Reading_ID_hemoglobin_hgb__human.append(i+"_hemoglobin(hgb)_human")
    Reading_ID_cholesterol_ldl_human.append(i+"_cholesterol_ldl_human'")



all_Reading_ID=[]
all_target=[]
for i in range(0,len(Reading_ID_hdl_cholesterol_human)):
    all_Reading_ID.append(Reading_ID_hdl_cholesterol_human[i])
    all_target.append(pred[i,0])
    all_Reading_ID.append(Reading_ID_hemoglobin_hgb__human[i])
    all_target.append(pred[i,1])
    all_Reading_ID.append(Reading_ID_cholesterol_ldl_human[i])
    all_target.append(pred[i,2])
    
dt=pd.DataFrame({
    'Reading_ID':all_Reading_ID,
    'target':all_target
    },index=None)
dt=dt.replace([1,2,3],unique)


dt.to_csv('Submission.csv', index = False)
# import pickle
# file="model.sav"
# pickle.dump(clf,open(file,"wb"))
