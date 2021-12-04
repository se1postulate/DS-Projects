# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 17:58:19 2021

@author: postulate-2
"""

import tensorflow as tf
import numpy as np
import pandas as pd 

import warnings
warnings.simplefilter('always')


data_csv=pd.read_csv('./corn_samples.csv',delimiter=',')
#print("sum of null values",data_csv.isnull().sum())
X= data_csv.drop('yield',1)




Model=tf.keras.models.load_model("my_model.h5")

year=int(input("Please Enter The Year.. "))
location=int(input("Please Enter The location between 0-41.. "))

#print(X.shape[1])
year_filter= X[X['year']==year]

location_filter=year_filter[year_filter['loc_ID']==location]

if(location_filter.shape[0]==0) :
    print("Details Not Found")
else:
    location_filter=np.array(location_filter)
    dt=location_filter.reshape((1,X.shape[1],1))
    predicted=Model.predict(dt)
    print("yield :",predicted[0][0])