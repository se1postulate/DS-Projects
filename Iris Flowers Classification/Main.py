# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:16:25 2021

@author: AMK
"""
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
iris=load_iris()
x=iris.data
y=iris.target
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.20,shuffle=True)
svc=SVC(gamma='scale',shrinking=False)
print(svc)
svc.fit(xtrain,ytrain)
pre=svc.predict(xtest)
print(confusion_matrix(pre,ytest))
print(classification_report(pre,ytest))
