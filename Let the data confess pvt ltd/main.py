# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:31:51 2021

@author: postulate-2
"""
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout, Flatten,TimeDistributed, MaxPooling1D,Conv1D, MaxPool2D, BatchNormalization,MaxPooling2D,Bidirectional
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,Flatten,GRU,TimeDistributed,Bidirectional
#load the data 
data=pd.read_csv("cancer_reg.csv",encoding= 'unicode_escape')
data_m=data
#Geography_u=data['Geography'].unique()

data['binnedInc']=data['binnedInc'].str.replace("]","")
data['binnedInc']=data['binnedInc'].str.replace("[","")
data['binnedInc']=data['binnedInc'].str.replace("(","")
data['binnedInc']=data['binnedInc'].str.replace("]","")

data[['binnedInc1','binnedInc2']]=data['binnedInc'].str.split(',', expand=True)

del data['binnedInc']

# all Geography have seperate values
del data['Geography']
#data['Geography']=data['Geography'].replace(Geography_u,np.arange(len(Geography_u)))

#change the data type  for all collumns as float
data.astype(float)

#find the sum of nan value
data.isnull().sum()

data=data.fillna(data.mean())


X=data.drop(columns=['TARGET_deathRate'])

Y=data['TARGET_deathRate']

X=np.array(X)
input_scaler =MinMaxScaler()
input_scaler =input_scaler .fit(X)
X=input_scaler .transform(X)
# X = np.expand_dims(X,axis=2)
# X=X.reshape(X.shape[1],X.shape[0],1)

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.05,shuffle=True)

xtrain = np.expand_dims(xtrain,axis=2)
xtest = np.expand_dims(xtest,axis=2)

input_shape_dim = (xtrain.shape[1],1)



def create_model(units, m):
    model = Sequential()
    model.add(m (units = units, return_sequences = True,
                input_shape = [xtrain.shape[1], xtrain.shape[2]]))
    model.add(Dropout(0.2))
    model.add(m (units = units))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    #Compile model
    model.compile(loss='mse', optimizer='adam',metrics=['mean_squared_error'])
    return model


model_gru = create_model(64, GRU)
model_lstm = create_model(64, LSTM)



model_gru.fit(xtrain, ytrain,batch_size=64,epochs=20,validation_data=(xtest, ytest))




# model = Sequential()
# # Adding a Bidirectional LSTM layer
# model.add(Bidirectional(LSTM(64,return_sequences=True, dropout=0.5, input_shape=(xtrain.shape[1], xtrain.shape[-1]))))
# model.add(Bidirectional(LSTM(20, dropout=0.5)))

# model.add(Dense(1))

# model.compile(loss='mse', optimizer='rmsprop',metrics=['mean_squared_error'])

# xtest=xtest.astype(float)
# ytest=ytest.astype(float)
# xtrain=xtrain.astype(float)
# ytrain=ytrain.astype(float)

# model.fit(xtrain, ytrain,batch_size=64,epochs=20,validation_data=(xtest, ytest))
