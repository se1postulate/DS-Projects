# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:18:33 2021

@author: asus
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

url="diabetes.csv"
sam = pd.read_csv(url)
data = pd.read_csv(url)
col=data.columns

unique=list(data[col[1]].unique())
uniqueT=list(data[col[4]].unique())
data[col[0]]=data[col[0]].replace(unique,[1,0])
data[col[1]]=data[col[1]].replace(unique,[1,0])
data[col[2]]=data[col[2]].replace(unique,[1,0])
data[col[3]]=data[col[3]].replace(unique,[1,0])
data[col[4]]=data[col[4]].replace(uniqueT,[1,0])



X=data.iloc[:,:4]
Y=data.iloc[:,4]
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.10,shuffle=True)

#
from sklearn import tree
clf=tree.DecisionTreeClassifier()
#from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier(n_estimators=180)
clf.fit(xtrain,ytrain)
pred=clf.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(pred,ytest))
print(classification_report(pred,ytest))

#file="model.sav"
#pickle.dump(clf,open(file,"wb"))
