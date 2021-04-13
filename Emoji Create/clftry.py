# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:31:57 2021

@author: AMK
"""


"""
Created on Wed Mar 10 18:21:33 2021

@author: AMK
"""
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np

import keras.backend as k
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras import regularizers

Base=os.path.abspath('Data/train')
BT=os.listdir(Base)
nclasses=len(BT)

def train_data():
    x=[]
    y=[]
    for i in BT:
        t=os.path.join(Base,i)
        for j in os.listdir(t):
            img=os.path.join(t,j)
            img = image.load_img(img, target_size=(48,48),color_mode="grayscale")
            img_array = image.img_to_array(img)
            y.append(BT.index(i))
            x.append(img_array)
            
    x=np.array(x).reshape(-1,48,48,1)
    x /=255
    y = keras.utils.to_categorical(y,nclasses)
    return x,y



def test_data():
    Base=os.path.abspath('Data/test')
    BT=os.listdir(Base)
    nclasses=len(BT)
    xt=[]
    yt=[]
    for i in BT:
        t=os.path.join(Base,i)
        for j in os.listdir(t):
            img=os.path.join(t,j)
            img = image.load_img(img, target_size=(48,48),color_mode="grayscale")
            img_array = image.img_to_array(img)
            img_array.shape
            yt.append(BT.index(i))
            xt.append(img_array)
    xt=np.array(xt).reshape(-1,48,48,1)    
    xt /=255    
    yt = keras.utils.to_categorical(yt,nclasses)
    return xt,yt


x,y=train_data()
xt,yt=test_data()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,shuffle=True)
import autokeras as ak
clf=ak.ImageClassifier()
clf.fit(xtrain,ytrain)
r=clf.predict(xtest)
