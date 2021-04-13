# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 12:29:52 2021

@author: AMK
"""

from keras.preprocessing.image import ImageDataGenerator
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



batch_size = 64

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'Data/train',  # this is the target directory
        target_size=(48,48),  # all images will be resized to 150x150
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    
       )  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(48,48),
        batch_size=batch_size
        ,color_mode="grayscale",
        class_mode="categorical",
    )



batch_size = 64
epochs = 4
input_shape=(48,48,1)
   
model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape,padding='same'))
model.add(MaxPooling2D((2, 2),padding='same'))

#model.add(Dropout(0.30))
#
#model.add(Conv2D(64, (3,3),activation='relu',padding='same'))
#model.add(MaxPooling2D((2, 2),padding='same'))
#model.add(Dropout(0.30))

#model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
#model.add(MaxPooling2D((2, 2),padding='same'))
#model.add(Dropout(0.30))  

##model.add(Conv2D(64, (3,3),activation='relu',padding='same'))
##model.add(MaxPooling2D((2, 2),padding='same'))
#
#
#model.add(Conv2D(32,(3,3),kernel_regularizer=regularizers.l1(0.015),activation='relu',padding='same'))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
  


model.load_weights("tryweights.h5")
model.fit_generator(train_generator,epochs=4 ,validation_data=validation_generator)
   


#  save the weights in h5 format
model.save_weights("tryweights.h5")


import tkinter as tk
from tkinter import filedialog

application_window = tk.Tk()

my_filetypes = [('all files', '.*'), ('text files', '.txt')]

img=filedialog.askopenfilename(parent=application_window,
                                    initialdir=os.getcwd(),
                                    title="Please select a file:",
                                    filetypes=my_filetypes)
img = image.load_img(img, target_size=(48,48),color_mode="grayscale")


img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array,axis=0)
img_batch /=255
pre=model.predict(img_batch)
import matplotlib.pyplot as plt
plt.imshow(img,cmap="gray")
predicted_classes = np.argmax((pre),axis=1)

print("predicted Result:",predicted_classes)
Base=os.path.abspath('Data/test')
BT=os.listdir(Base)
print("predicted Result:",BT[predicted_classes[0]])
application_window.destroy()