from nautilus_trader.indicators.base.indicator import Indicator
from nautilus_trader.model.data import Bar

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import rsi
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from ta.trend import SMAIndicator

from agent import CustomAgent
from tensorflow.keras.optimizers import Adam
from collections import deque

import json
import pandas as pd


class ModelIndicator(Indicator):

    def __init__(self,
                 numBars: int,
                 numPrediction: int):
        super().__init__([numBars, numPrediction])

        with open("models/2024_11_07_22_40/" + "/Parameters.json", "r") as json_file:
            params = json.load(json_file)
        params["Actor name"] = f"latest_eps350_Actor.keras"
        params["Critic name"] = f"latest_eps350_Critic.keras"

        self.model =  CustomAgent(lookback_window_size=params["lookback window size"], optimizer=Adam, depth=params["depth"])

        self.numPrediction = numPrediction
        self.candleData = deque(maxlen=numBars)
        self.df = pd.DataFrame(columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.numBars = numBars

    def handle_bar(self, bar: Bar):
        # Append the new bar data to df
        new_row = {
            'Timestamp': bar.ts_event,
            'Open': bar.open.as_double(),
            'High': bar.high.as_double(),
            'Low': bar.low.as_double(),
            'Close': bar.close.as_double(),
            'Volume': bar.volume.as_double()  # Assuming you have volume in the Bar
        }
        new_row_df = pd.DataFrame([new_row])

        # Concatenate the new row DataFrame with the existing DataFrame
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)

        # Recalculate indicators
        if len(self.df) > 14:
            self.calculate_indicators()
            self.normalize_data()
            
            recent_data = self.df.iloc[-1]              
            self.candleData.append(dict(
                # Timestamp=recent_data['Timestamp'],
                Close=recent_data['norm_close'],
                rsi_5=recent_data['rsi_5'],
                rsi_7=recent_data['rsi_7'],
                cmf=recent_data['cmf'],
                natr=recent_data['natr'],
                sma_7=recent_data['sma_7'],
            ))
       
            if len(self.candleData) == self.numBars:
                self.prediction = self.predict()
                self._set_initialized(True)
        
        self.prediction = None


    def calculate_indicators(self):
        self.df["sma_7"] = SMAIndicator(close=self.df["Close"], window=7, fillna=True).sma_indicator()

        # Add RSI
        self.df["rsi_5"] = rsi(close=self.df["Close"], window=5, fillna=True)
        self.df["rsi_7"] = rsi(close=self.df["Close"], window=7, fillna=True)

        # Add Average True Range
        self.df["atr"] = AverageTrueRange(high=self.df["High"], low=self.df["Low"], close=self.df["Close"], window=7, fillna=True).average_true_range()

        # Add On-balance Volume
        self.df["cmf"] = ChaikinMoneyFlowIndicator(high=self.df["High"], low=self.df["Low"], close=self.df["Close"], volume=self.df["Volume"], window=14, fillna=True).chaikin_money_flow()

        return self.df
    
    def normalize_data(self):

        # Logging and Differencing
        self.df['natr'] = self.df['atr'] / self.df['Close']
        self.df['rsi_5'] = self.df['rsi_5'] / 100
        self.df['rsi_7'] = self.df['rsi_7'] / 100

        sma_scaler  = MinMaxScaler()
        self.df['sma_7'] = np.log(self.df['sma_7']) - np.log(self.df['sma_7'].shift(1))
        self.df[['sma_7']] = sma_scaler.fit_transform(self.df[['sma_7']])
        
        close_scaler = MinMaxScaler()
        self.df['norm_close'] = np.log(self.df['Close']) - np.log(self.df['Close'].shift(1))
        self.df[['norm_close']] = sma_scaler.fit_transform(self.df[['norm_close']])

        # Keep only relevant columns
        self.df = self.df.dropna()  # dropna to remove any NaN values from shifting


    def isRising(self, col):
        if self.prediction is None:
            self.prediction = self.predict()
        return (self.prediction[col] > 0).all()
    
    def isFalling(self, col):
        if self.prediction is None:
            self.prediction = self.predict()
        return (self.prediction[col] < 0).all()

    def predict(self):
        data = pd.DataFrame(self.candleData)
        data = data[['Close', 'rsi_5', 'rsi_7', 'cmf', 'natr', 'sma_7']]
        action, prediction = self.model.act(data)
        self.prediction = action
        return action
       
    def _reset(self):
        self.candleData.clear()
