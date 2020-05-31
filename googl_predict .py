import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class windowsGOOGLE:
 df = web.DataReader('GOOGL', data_source='yahoo', start='2012-01-01', end='2020-03-07')
##df

 plt.figure(figsize=(16,8))
 plt.title('Google')
 plt.plot(df['Close'])
 plt.xlabel('Date', fontsize=18)
 plt.ylabel('Cena v USD($)', fontsize=18)
 plt.show()

 data =df.filter(['Close'])
 dataset = data.values
 training_data_len = math.ceil(len(dataset)* .8)

##training_data_len

 scaler = MinMaxScaler(feature_range=(0,1))
 scaled_data = scaler.fit_transform(dataset)

##scaled_data

 train_data = scaled_data[0:training_data_len, :]

 x_train = []
 y_train = []


 for i in range(60, len(train_data)):
     x_train.append(train_data[i-60:i, 0])
     y_train.append(train_data[i,0])
     if i<= 61:
         print(x_train)
         print(y_train)

 x_train, y_train = np.array(x_train), np.array(y_train)
 x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

 model = Sequential()
 model.add(LSTM(50, return_sequences= True, input_shape=(x_train.shape[1], 1)))
 model.add(LSTM(50, return_sequences= False))
 model.add(Dense(25))
 model.add(Dense(1))

 model.compile(optimizer='adam',loss='mean_squared_error')

 model.fit(x_train, y_train, batch_size=1, epochs=1)


 test_data = scaled_data[training_data_len - 60: , :]
 x_test = []
 y_test = dataset[training_data_len: , :]

 for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
 x_test = np.array(x_test)

 x_test =np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

 predictions = model.predict(x_test)
 predictions = scaler.inverse_transform(predictions)

 rmse = np.sqrt( np.mean(predictions - y_test)**2)
# rmse

##narisan graf z uporabljenimi podatki in uporaba za treniranje
 train = data[:training_data_len]
 valid = data[training_data_len:]
 valid['Predictions'] = predictions
##narisanje predvidevanja
 plt.figure(figsize=(16,8))
 plt.title('Predvidevana vrednost')
 plt.xlabel('Datum', fontsize=18)
 plt.ylabel('Cena v USD($)', fontsize=18)
 plt.plot(train['Close'])
 plt.plot(valid[['Close', 'Predictions']])
 plt.legend(['Treniranje', 'Vrednost','Predvidevanje'], loc='lower right')
 plt.show()

 valid 
##predvidevanje na tocen dolocen datum 
 googl_quote = web.DataReader('GOOGL', data_source='yahoo', start='2012-01-01', end='2020-03-10')

 new_df = googl_quote.filter(['Close'])
 last_60_days = new_df[-60:].values

 last_60_days_scaled = scaler.transform(last_60_days)

 X_test = []
 X_test.append(last_60_days_scaled)
#convert to numpy array
 X_test = np.array(X_test)
 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
 pred_price = model.predict(X_test)
 pred_price = scaler.inverse_transform(pred_price)
 print("Predvidena vrednost za datum", pred_price)

