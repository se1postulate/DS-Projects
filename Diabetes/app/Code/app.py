# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:31:34 2021

@author: AMK
"""


import pandas as pd
from joblib import load 
clf=load("model.joblib")

while True:
    f=input("Father have diabetes? \n")
    if f=="s" or  f=="S" or f=="yes" or f=="Yes" or f=="YES":
        f=1
    else:
        f=0
    m=input("Mother have diabetes? \n")
    if m=="s" or  m=="S" or m=="yes" or m=="Yes" or m=="YES":
        m=1
    else:
        m=0
    fh=input("Father's Heredity have diabetes? \n")
    if fh=="s" or  fh=="S" or fh=="yes" or fh=="Yes" or fh=="YES":
        fh=1
    else:
        fh=0
    mh=input("Father's Heredity have diabetes?  \n")
    if mh=="s" or  mh=="S" or mh=="yes" or mh=="Yes" or mh=="YES":
        mh=1
    else:
        mh=0
    d=[f,m,fh,mh]
    result=clf.predict([d])
    if result==1:
        print("child have diabetes \n")
    else:
        print("child dont have diabetes \n")
    c=input("do you want continue?  \n")
    if c=="N" or c=="n" or c=="No" or c=="no" or c=="NO":
        break
    
