# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:41:41 2021

@author: postulate-41
"""

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


ori_path=os.listdir('output/')
N=len(os.listdir('output'))

X=[]
Y=[]
cl=[]

for i in range(N):
#    print(i)
    cl.append(ori_path[i])
    path1=f"output/{ori_path[i]}"
    
    list1=os.listdir(path1)
    k=0
    for j in list1:
        path2=f"{path1}/{j}"
        image=cv2.imread(path2,0)
        X.append(image)
        Y.append(i)
        k=k+1
        if k==1000:
            break
        
        
X=np.array(X)
Y=np.array(Y)
#
#np.save("X",X)
#np.save("Y",Y)

Y=tf.keras.utils.to_categorical(Y)

X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.03,shuffle=True,random_state=True)


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(30,30, 1),padding="same"))
model.add(layers.MaxPooling2D((2, 2),padding='same'))

model.add(layers.Dropout(0.30))

model.add(layers.Conv2D(64, (3,3),activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2, 2),padding='same'))
model.add(layers.Dropout(0.30))

model.add(layers.Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(layers.MaxPooling2D((2, 2),padding='same'))
model.add(layers.Dropout(0.30))  

#model.add(Conv2D(64, (3,3),activation='relu',padding='same'))
#model.add(MaxPooling2D((2, 2),padding='same'))

model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6,activation="softmax"))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True),
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model2.h5',monitor='val_accuracy',
                             mode='max',
                             save_best_only=True,
                             verbose=1),
]



#model = tf.keras.models.load_model('best_model.h5')
model.fit(xtrain, ytrain, epochs=10, 
                    validation_data=(xtest, ytest),callbacks=my_callbacks)

# model.save("model.h5")

