# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 19:10:28 2021

@author: AMK
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv("Data/data.csv")
def encode(data):
    data=data.drop(["Id","Alley","PoolQC","Fence","MiscFeature"],axis=1)
    data=data.fillna(0)
    dsc=[]#String data type column
    dc=data.columns#data column
    
    for col in data.select_dtypes(include=['object']).columns:
        dsc.append(col)
    df={}
    du={}
    for i in dc:
        if i in dsc:
            uv=data[i].unique() 
            
            v=len(uv)
            d=data[i].replace(uv,np.arange(1,v+1))
            df[i]=d
            du[i]=uv
        else:
            d=data[i]
            df[i]=d
    data2=pd.DataFrame(df)
    
    return data2,du
              
data2,du=encode(data)
x=data2.drop(['SalePrice'],axis=1)
y=data2['SalePrice']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30,shuffle=True)

reg=RandomForestRegressor(n_estimators=100)
reg.fit(xtrain,ytrain)
pred=reg.predict(xtest)
print(reg.score(xtest,ytest))
#print(du)

def outputencode(data,du):
    data=data.drop(["Id","Alley","PoolQC","Fence","MiscFeature"],axis=1)
    data=data.fillna(0)
    dsc=[]#String data type column
    dc=data.columns#data column
    for col in data.select_dtypes(include=['object']).columns:
        dsc.append(col)
    df={}
    
    for i in dc:
        if i in dsc:
            uv=list(du[i])
            v=len(uv)
            d=data[i].replace(uv,np.arange(1,v+1))
            df[i]=d
        else:
            d=data[i]
            df[i]=d
    data2=pd.DataFrame(df)
    return data2

datap=pd.read_csv("Data/test.csv") 
datap=outputencode(datap,du)  
pre=reg.predict(datap)

    


