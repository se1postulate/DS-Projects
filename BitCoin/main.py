import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import Dense, Dropout,GRU
from keras import optimizers 
from sklearn.preprocessing import MinMaxScaler

seed = 1234
np.random.seed(seed)
plt.style.use('ggplot')
data = pd.read_csv('F:\pythonlearn\Data Science\Exercise\Practice\BitCoin\coin.csv',index_col='Date', parse_dates=['Date'])
data = pd.DataFrame(data['Close'])
dataw=data
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

lag = 2
# Sliding windows function
def create_sliding_windows(data,len_data,lag):
    x=[]
    y=[]
    for i in range(lag,len_data):
        x.append(data[i-lag:i,0])
        y.append(data[i,0]) 
    return np.array(x),np.array(y)

datas=np.array(data)

x, y = create_sliding_windows(datas,len(datas), lag)
x = np.reshape(x, (x.shape[0],x.shape[1],1))
 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,shuffle=False)
xtest,xval,ytest,yval=train_test_split(xtest,ytest,test_size=0.30,shuffle=False)

learning_rate = 0.0001
hidden_unit = 64
batch_size=100
epoch =100

# Architecture Gated Recurrent Unit
regressorGRU = Sequential()
# First GRU layer with dropout
regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(xtrain.shape[1],1), activation = 'tanh'))
regressorGRU.add(Dropout(0.2))
# Second GRU layer with dropout
regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, activation = 'tanh'))
regressorGRU.add(Dropout(0.2))
# Third GRU layer with dropout
regressorGRU.add(GRU(units=hidden_unit, return_sequences=False, activation = 'tanh'))
regressorGRU.add(Dropout(0.2))

# Output layer
regressorGRU.add(Dense(units=1))


regressorGRU.compile(optimizer=optimizers.Adam(lr=learning_rate),loss='mean_squared_error')

#from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
# Fitting ke data training dan data validation

pred = regressorGRU.fit(xtrain, ytrain, validation_data=(xval,yval),batch_size=batch_size, epochs=epoch)


y_pred_test = regressorGRU.predict(xtest)

ypred = scaler.inverse_transform(y_pred_test)
ypred = np.reshape(ypred, (ypred.shape[0]))




ytest = np.reshape(ytest, (ytest.shape[0],1))

ytest = scaler.inverse_transform(ytest)
ytest = np.reshape(ytest, (ytest.shape[0]))


datacompare = pd.DataFrame()
datacompare['Data Test'] = ytest

datacompare['Prediction Results'] = ypred
datacompare


plt.figure(num=None, figsize=(10, 4), dpi=80,facecolor='w', edgecolor='k')
plt.title('Graph Comparison Data Actual and Data Prediction')
plt.plot(datacompare['Data Test'], color='red',label='Data Test')
plt.plot(datacompare['Prediction Results'], color='blue',label='Prediction Results')
plt.xlabel('Day')
plt.ylabel('Price')
plt.legend()
plt.show()


import math
MSE = np.square(np.subtract(ytest,ypred)).mean()
RMSE = math.sqrt(MSE)
print('root mean squad error',RMSE)