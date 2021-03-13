# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:15:20 2021

@author: AMK

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('news/news.csv')

x_train,x_test,y_train,y_test=train_test_split(df['text'],df['label'], test_size=0.2, random_state=7)

tf=TfidfVectorizer(stop_words='english', max_df=0.7)
train=tf.fit_transform(x_train)
test=tf.transform(x_test)
print(test.shape)

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(train,y_train)
pre=pac.predict(test)
print(accuracy_score(pre,y_test))

