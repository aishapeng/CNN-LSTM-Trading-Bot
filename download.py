# IMPORTS
import pandas as pd
import math
import os.path
from datetime import datetime
import random
# from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
import numpy as np
from indicators import *
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, TimeDistributed
from tensorflow.keras import backend as K


# from tqdm import tqdm_notebook  # (Optional, used for progress-bars)

### API
binance_api_key = '8TYEY8hkBMeYE53NSLUbaPlud0k4sJdFTfrJ7JUcW0NupkC5VZwrOu2KKXxYbg0N'  # Enter your own API-key here
binance_api_secret = 'UVZN3VV4IWvi4xjS5s1AHHxXWGrTcUd5oUeNeAx4KPVMzf3GEvElgkpzJm8yeONj'  # Enter your own API-secret here

# binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
# batch_size = 750
# # bitmex_client = bitmex(test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)
# binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### FUNCTIONS
# def minutes_of_new_data(symbol, kline_size, data, source):
#     if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
#     elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
#     # elif source == "bitmex": old = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=False).result()[0][0]['timestamp']
#     if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
#     # if source == "bitmex": new = bitmex_client.Trade.Trade_getBucketed(symbol=symbol, binSize=kline_size, count=1, reverse=True).result()[0][0]['timestamp']
#     return old, new
#
# def get_all_binance(symbol, kline_size, save = False):
#     filename = '%s-%s-data.csv' % (symbol, kline_size)
#     if os.path.isfile(filename): data_df = pd.read_csv(filename)
#     else: data_df = pd.DataFrame()
#     oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
#     delta_min = (newest_point - oldest_point).total_seconds()/60
#     available_data = math.ceil(delta_min/binsizes[kline_size])
#     if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
#     else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
#     klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
#     data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
#     data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
#     if len(data_df) > 0:
#         temp_df = pd.DataFrame(data)
#         data_df = data_df.append(temp_df)
#     else: data_df = data
#     data_df.set_index('timestamp', inplace=True)
#     if save: data_df.to_csv(filename)
#     print('All caught up..!')
#     return data_df

if __name__ == "__main__":
    df = pd.read_csv('./BTCUSDT_cycle1.csv')
    # df = df.dropna()
    #
    # df = df.rename(columns={'time': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    #                         'volume': 'Volume', 'trades': 'Trades'})
    # # df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
    # # df = df.sort_values('Timestamp')
    #
    # df = AddIndicators(df)  # insert indicators
    # df = df[100:].dropna()  # cut first 100 in case for indicator calc
    # # df = df.iloc[:, 7:]
    # # print(df)
    # # depth = len(list(df.columns[1:]))  # OHCL + indicators without Date
    # df = df.reset_index(drop=True)
    market_history =[]
    a= []
    for i in reversed(range(10)):
        market_history.append([df.loc[i, column] for column in df.columns])

    o = np.expand_dims(market_history, axis=0) #(1, 10, 9) axis = 0 : (10, 1 , 9) axis =1
    o = np.vstack(market_history) #(10, 9)
    # o = np.array(market_history)
    # o = np.expand_dims(o, axis=1)
    #
    print(o)
    print(o.shape)
    # c = (o, np.zeros((o.shape[0], 1)))
    # print(c)
    # tf1 = tf.convert_to_tensor(o)
    # print(tf1)
    # print(tf1.shape)
    # # a.append(o)
    # # print(a)
    # # print(a)
    # # a = np.array(market_history)
    # # a = a[:, np.newaxis]
    #
    # market_history.append([df.loc[55, column] for column in df.columns])
    # X_input = Input((100,8))
    # X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X_input) # 100 rows, 64 features
    # X = MaxPooling1D(pool_size=2)(X) # 50 rows, 64 features
    # # X = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(X)  # 50 rows, 32 features
    # # X = MaxPooling1D(pool_size=2)(X)  # 25 rows, 32 features
    # # print(X[0])
    # X = LSTM(32, return_sequences=True, input_shape=(1, X[1], X[2]))(X)
    # X = Flatten()(X)
    #
    #
    # # V = Dense(64, activation="relu")(X)
    # # V = Dense(32, activation="relu")(X)
    # # value = Dense(1, activation=None)(V)
    #
    # # Actor model
    # # A = Dense(64, activation="relu")(X)
    # A = Dense(32, activation="relu")(X)
    # output = Dense(3, activation="softmax")(A)
    #
    # Actor = Model(inputs=X_input, outputs=output)
    # # self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=lr))
    # print(Actor.summary())