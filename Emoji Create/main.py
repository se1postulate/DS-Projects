# -*- coding: utf-8 -*-
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

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    fill_mode='nearest',
    validation_split = 0.2
    )

datagen.fit(xtrain)

train_generator = datagen.flow(xtrain, ytrain, batch_size=60, subset='training')

validation_generator = datagen.flow(xtrain, ytrain, batch_size=60, subset='validation')



batch_size = 64
epochs = 4
num_classes = nclasses
input_shape=(48,48,1)
   
model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape,padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.30))

model.add(Conv2D(64, (3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.30))

model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.30))  

##model.add(Conv2D(64, (3,3),activation='relu',padding='same'))
##model.add(MaxPooling2D((2, 2),padding='same'))
#
#
#model.add(Conv2D(32,(3,3),kernel_regularizer=regularizers.l1(0.015),activation='relu',padding='same'))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nclasses, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


## load weights
model.load_weights("model.h5")

model.fit(xtrain,ytrain, batch_size=100,epochs=1,verbose=1,validation_data=(xtest, ytest))


#  save the weights in h5 format
#model.save_weights("model.h5")


accuracy = model.evaluate(xt,yt,batch_size=100)
print(accuracy[1])

#
#
#



#
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)



img="nat.jpg"
img = image.load_img(img, target_size=(48,48),color_mode="grayscale")


img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array,axis=0)
img_batch /=255
pre=model.predict(img_batch)
import matplotlib.pyplot as plt
plt.imshow(img,cmap="gray")
predicted_classes = np.argmax((pre),axis=1)

print("predicted Result:",BT[predicted_classes[0]].upper())
