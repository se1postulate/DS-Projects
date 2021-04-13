# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 09:18:33 2021

@author: AMK
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller
data=pd.read_csv('Data/time.csv')
data.columns=['month','sales']
data.shape
data.isnull().sum()
data.tail()
data.drop([105,106],axis=0,inplace=True)
data['month']=pd.to_datetime(data['month'])
data.set_index('month',inplace=True)


def adfuller_test(sales):
    result=adfuller(sales)
    labels=['adf test','pvalue','lagsused','number of obs used']
    for value,label in zip(result,labels):
        print('{}:{}'.format(label,value))
    if result[1]<=0.05:
        print('strong evidance against null hypo,reject the null hypo & data is stationary')
    else:
        print('strong evidance against null hypo,accept the null hypo & data is not stationary')
        
adfuller_test(data['sales'])
data['sales'].shift(1)
data['sales first difference']=data['sales']-data['sales'].shift(1)
data['seasonal first diff']=data['sales']-data['sales'].shift(12)
adfuller_test(data['seasonal first diff'].dropna())
data.plot()

from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=plot_acf(data['seasonal first diff'].iloc[13:],lags=40,ax=ax1)
ax2=fig.add_subplot(212)
fig=plot_pacf(data['seasonal first diff'].iloc[13:],lags=40,ax=ax2)

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(data['sales'],order=(1,1,1))
model=model.fit()
model.summary()
data['arima']=model.predict(start=90,end=103,dynamic=True)


import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(data['sales'],order=(1,0,1),seasonal_order=(1,1,1,12))
results=model.fit()
data['sari']=results.predict(start=12,end=103,dynamic=True)
data[['sales','sari','arima'   ]].plot(figsize=(12,10))


from pandas.tseries.offsets import DateOffset

future_dates=[data.index[-1]+DateOffset(months=x)for x in range(0,24)]
fdata=pd.DataFrame(index=future_dates[1:],columns=data.columns)
ndata=pd.concat([data,fdata])
ndata['fore']=results.predict(start=102,end=126,dynamic=True)
ndata[['sales','sari','fore']].plot(figsize=(12,10))
