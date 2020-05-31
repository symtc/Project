import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import quandl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
quandl.ApiConfig.api_key = '9Mh_7A3tmG1cu8cJ9XX7'

class windowTESLA:
 df = web.DataReader('TSLA', data_source='yahoo', start='2011-01-01', end='2020-05-28') 
#pobere podatke iz yahoo financial  jih uporabi  za učenje
##df

 #plt.figure(figsize=(16,8))
 #plt.title('Tesla')
 #plt.plot(df['Close'])
 #plt.xlabel('Date', fontsize=18)
 #plt.ylabel('Cena v USD($)', fontsize=18)
 #plt.show()

#katere podatke vzame 
 data =df.filter(['Close'])
 dataset = data.values
 
 training_data_len = math.ceil(len(dataset)* .8)

##training_data_len

 scaler = MinMaxScaler(feature_range=(0,1))
 scaled_data = scaler.fit_transform(dataset)

##scaled_data izpiše podana podatki ki si jih je izpisal

 train_data = scaled_data[0:training_data_len, :]

 x_train = []
 y_train = []

#treniranje x,y, z izpisom x,y osi
 for i in range(60, len(train_data)):
     x_train.append(train_data[i-60:i, 0])
     y_train.append(train_data[i,0])
     if i<= 61:
         print(x_train)
         print(y_train)

 x_train, y_train = np.array(x_train), np.array(y_train)
#izračun x,y osi na grafu
 x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))



 model = Sequential()
 model.add(LSTM(50, return_sequences= True, input_shape=(x_train.shape[1], 1)))
 model.add(LSTM(50, return_sequences= False))
 model.add(Dense(25))
 model.add(Dense(1))

 model.compile(optimizer='adam',loss='mean_squared_error')

 model.fit(x_train, y_train, batch_size=1, epochs=1)

#test podatki in izpis podatkov
 test_data = scaled_data[training_data_len - 60: , :]
 x_test = []
 y_test = dataset[training_data_len: , :]

 for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
 x_test = np.array(x_test)

 x_test =np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
#napoved in izris v tabeli
 predictions = model.predict(x_test)
 predictions = scaler.inverse_transform(predictions)


#root mean square da so podatki v normalnem grafu
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

 #valid 
##predvidevanje na tocen dolocen datum v prihodnosti
 tsla_quote = web.DataReader('TSLA', data_source='yahoo', start='2011-01-01', end='2020-05-30')

 new_df = tsla_quote.filter(['Close'])
 last_60_days = new_df[-60:].values

 last_60_days_scaled = scaler.transform(last_60_days)

 X_test = []
 X_test.append(last_60_days_scaled)
#convert to numpy array

 X_test = np.array(X_test)
 X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
 pred_price = model.predict(X_test)
 pred_price = scaler.inverse_transform(pred_price)




 #Predvidevanje za 1 dan
 print("Predvidena vrednost za datum 30.05.2020 ", pred_price)
 




 #Predvidevanje za 7 dni
 df = quandl.get("WIKI/TSLA")
 df = df[['Adj. Close']]
 forecast_out = 7
 df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
 #print(df.tail())
 X = np.array(df.drop(['Prediction'],1))
 X = X[:-forecast_out]
 ##print(X)
 
 y = np.array(df['Prediction'])
 y = y[:-forecast_out]
 #print(y)
 x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
 svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
 svr_rbf.fit(x_train, y_train)
 
 svm_confidence = svr_rbf.score(x_train, y_train)
 #print("svm confidence", svm_confidence)
 
 lr = LinearRegression()
 lr.fit(x_train, y_train) 
  
 lr_confidence = lr.score(x_train, y_train)
 ##print("lr confidence", lr_confidence)
 
 x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
 
 print("Napoved za 7 dni")
 lr_prediction =lr.predict(x_forecast)
 print(lr_prediction)
 #support vector
 svm_prediction =svr_rbf.predict(x_forecast)
 print(svm_prediction)
 




 
 #Predvidevanje za 30 dni
 df = quandl.get("WIKI/TSLA")
 df = df[['Adj. Close']]
 forecast_out = 30
 df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
 #print(df.tail())
 X = np.array(df.drop(['Prediction'],1))
 X = X[:-forecast_out]
 ##print(X)
 
 y = np.array(df['Prediction'])
 y = y[:-forecast_out]
 ##print(y)
 x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 
 svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
 svr_rbf.fit(x_train, y_train)
 
 svm_confidence = svr_rbf.score(x_train, y_train)
 ##print("svm confidence", svm_confidence)
 
 lr = LinearRegression()
 lr.fit(x_train, y_train) 
  
 lr_confidence = lr.score(x_train, y_train)
 ##print("lr confidence", lr_confidence)
 
 x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
 
 print("Napoved za 30 dni")
 lr_prediction =lr.predict(x_forecast)
 print(lr_prediction)
 ##support vector
 svm_prediction =svr_rbf.predict(x_forecast)
 print(svm_prediction)
 
 
 

