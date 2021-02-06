"""
Created on Tue Feb  2 13:17:03 2021
@author: AMK
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.utils import resample
dataset=pd.read_excel('Data/default of credit card clients.xls')

df_0=dataset[dataset.Y==0]
df_1=dataset[dataset.Y==1]
df_1_new = resample(df_1,replace=True,n_samples=(len(df_0.index))) 
dataset2=pd.concat([df_0,df_1_new])


X=dataset2.iloc[3:,1:-1]
Y=dataset2.iloc[3:,-1]

xtrain,xtest,ytrain,ytest = train_test_split(X,Y, test_size=.05,shuffle=True,random_state=70)


scalar=StandardScaler()
scalar.fit(xtrain)
xtrain=scalar.transform(xtrain)
xtest=scalar.transform(xtest)
model=KNeighborsRegressor(n_neighbors=1)
model.fit(xtrain,ytrain)


pred=model.predict(xtest)

score=model.score(xtest,ytest)
print("score:",score)


print("Mean squad Error is :",format(np.power(pred,4).mean()))


score=r2_score(ytest,pred)
print("R2 score:",score)


pred=pd.DataFrame({'predict':pred,'ytest':ytest})
print(pred)
