# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:22:05 2021

@author: AMK
"""

import pandas as pd
from data import encode
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv("Data/data.csv")
data=data.dropna(axis=0)
data=encode(data)
df_1=data[data.Loan_Status==1]

df_0=data[data.Loan_Status==0]
df_0= resample(df_0,replace=True,n_samples=(len(df_1)))
print(len(df_0))
print(len(df_1))

data=pd.concat([df_0,df_1])
x=data.iloc[:,:11];
y=data.iloc[:,11];
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.10,shuffle=True)

print("************************SVC************************************")
svc=SVC(gamma='scale',shrinking=False)
print(svc)
svc.fit(xtrain,ytrain)
pre=svc.predict(xtest)
print(confusion_matrix(pre,ytest))
print(classification_report(pre,ytest))

print("************************KNN************************************")
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain,ytrain)
pre2=knn.predict(xtest)
print(confusion_matrix(pre2,ytest))
print(classification_report(pre2,ytest))

print("************************DecisionTree************************************")
tclf=DecisionTreeClassifier()
tclf.fit(xtrain,ytrain)
pre3=tclf.predict(xtest)
print(confusion_matrix(pre3,ytest))
print(classification_report(pre3,ytest))


print("************************Random Forest************************************")

rclf=RandomForestClassifier(n_estimators=150)
rclf.fit(xtrain,ytrain)
pre4=rclf.predict(xtest)
print(confusion_matrix(pre4,ytest))
print(classification_report(pre4,ytest))
