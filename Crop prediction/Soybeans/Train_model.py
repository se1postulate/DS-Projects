# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 16:15:45 2021

@author: postulate-2
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

from sklearn.metrics import mean_squared_error, r2_score


#MODEL LAYERS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,Dropout, Flatten,TimeDistributed, MaxPooling1D,Conv1D, MaxPool2D, BatchNormalization,MaxPooling2D,Bidirectional


#from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,BatchNormalization,\
#                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN,\
#LSTM, GlobalAveragePooling2D, SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D,Reshape,\
#Conv2DTranspose, LeakyReLU, Conv1D, AveragePooling1D, MaxPooling1D, MaxPool1D, GlobalAvgPool1D
#


data_csv=pd.read_csv('./soybean_samples.csv',delimiter=',')
#print("sum of null values",data_csv.isnull().sum())
Y=data_csv['yield']
X= data_csv.drop('yield',1)

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,train_size=0.98)


xtrain = np.expand_dims(xtrain,axis=2)
xtest = np.expand_dims(xtest,axis=2)

input_shape_dim = (xtrain.shape[1],1)

Early_Stopper = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")
Checkpoint_Model = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      filepath="./modelcheck")



Model = Sequential()

Model.add(Conv1D(256,5,strides=1,padding="same",activation="relu",input_shape=input_shape_dim))
Model.add(BatchNormalization())
Model.add(MaxPooling1D(3,strides=2,padding="same"))

Model.add(Conv1D(256,4,strides=1,padding="same",activation="relu"))
Model.add(Dropout(0.3))
Model.add(MaxPooling1D(3,strides=2,padding="same"))

Model.add(Conv1D(128,4,strides=1,padding="same",activation="relu"))
Model.add(Dropout(0.3))
Model.add(MaxPooling1D(3,strides=2,padding="same"))


#Model.add(Conv1D(64,4,strides=1,padding="same",activation="relu"))
#Model.add(Dropout(0.3))
#Model.add(MaxPooling1D(3,strides=2,padding="same"))
#
#Model.add(Conv1D(32,4,strides=1,padding="same",activation="relu"))
#Model.add(Dropout(0.3))
#Model.add(MaxPooling1D(3,strides=2,padding="same"))


#Model.add(Flatten())
Model.add(TimeDistributed(Flatten()))
Model.add(Bidirectional(LSTM(64,
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5)))
Model.add(LSTM(64,
                                  dropout=0.5,
                                  recurrent_dropout=0.5))
Model.add(Flatten())
Model.add(Dense(512, activation='relu'))

Model.add(Dense(1, activation='linear'))


Model.compile(optimizer="Adam",loss="mean_absolute_error",metrics=["mae"])

#
Model.fit(xtrain, ytrain,batch_size=64,epochs=5,validation_data=(xtest, ytest))
#
#
#Model.save("my_model.h5")
