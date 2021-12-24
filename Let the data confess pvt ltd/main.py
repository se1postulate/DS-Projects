# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:31:51 2021

@author: postulate-2
"""
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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

def create_model_bilstm(units):
    model = Sequential()
    model.add(Bidirectional(LSTM(units = units,                             
              return_sequences=True),
              input_shape=(xtrain.shape[1], xtrain.shape[2])))
    model.add(Bidirectional(LSTM(units = units)))
    model.add(Dense(1))
    #Compile model
    model.compile(loss='mse', optimizer='adam',metrics=['mean_squared_error'])
    return model

def fit_model(model):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                               patience = 10)
    history = model.fit(xtrain, ytrain, epochs = 100,  
                        validation_split = 0.2, batch_size = 32, 
                        shuffle = False, callbacks = [early_stop])
    return history


# GRU and LSTM
model_gru = create_model(64, GRU)
model_lstm = create_model(64, LSTM)


# BiLSTM
model_bilstm = create_model_bilstm(64)

#compare to GRU AND LSTM BiLSTM Gives best result


history_bilstm = fit_model(model_bilstm)
history_lstm = fit_model(model_lstm)
history_gru = fit_model(model_gru)

# Plot train loss and validation loss
def plot_loss (history):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
plot_loss (history_bilstm)
plot_loss (history_lstm)
plot_loss (history_gru)

